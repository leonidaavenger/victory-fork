//! Helper threads for the submitting thread's own work: the parts of a
//! submission's preparing that can be done side by side with the rest of
//! it (`crate::host`: the copies of a flush of refreshes, cut into pieces,
//! `VICTORY_REFRESH_PIECES`; the pass over the title's view heaps beside
//! the rest of the preparing, `VICTORY_HEAP_SYNC_BESIDE`).
//!
//! Why a crew of its own and not `crate::shadow`'s `wide`: the submitting
//! thread is the frame's limit (profile M12: 49.3 ms a frame under
//! `Host::submit`; the card idles 15-18 ms a frame and the title's render
//! thread stands 39-40 ms in its present waiting for it), while the process
//! keeps about 3 of the PC's 24 logical processors busy. The threads of
//! `wide` sleep on a condition variable after every job, and run wherever
//! the system puts them. A wake costs tens of microseconds, and a flush of
//! refreshes comes every few dozen microseconds; a slow core copies at
//! about half the speed of a fast one.
//!
//! Here:
//! - the helpers ask to run on the fast cores
//!   (`crate::framelog::keep_on_fast_cores`);
//! - a helper that runs out of jobs looks for more for a while ([`spin`]:
//!   500 microseconds unless `VICTORY_CREW_SPIN_US` says otherwise) before
//!   it sleeps, so that the stages of one submission find it awake;
//! - a job given to the crew wakes at most one sleeping helper, and only
//!   if one sleeps: a stage of many jobs wakes each helper once at most;
//! - the thread that waits for a group of jobs does its jobs too
//!   ([`Group::wait`]): the submitting thread takes its share.
//!
//! `VICTORY_CREW_THREADS` (0 to 16) says how many helpers there are: by
//! default a quarter of the logical processors, 1 to 5 (5 on this PC). With
//! 0, or if no helper could be started, every job runs on the thread that
//! gives it, at once. No helper is started before a switch that uses the
//! crew gives it its first job. The helpers' names begin `victory-wide-`,
//! so that the sampling profile counts them with the other helpers.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

type Work = Box<dyn FnOnce() + Send>;

/// What the jobs of one group share: how many of them are not done, and
/// the first panic one of them ended with (given on to the thread that
/// waits for the group).
#[derive(Default)]
struct Tally {
    left: AtomicUsize,
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

struct Job {
    tally: Arc<Tally>,
    work: Work,
}

/// One helper: whether it sleeps (set by itself, taken by who wakes it),
/// and its thread, to wake it with.
struct Hand {
    sleeping: AtomicBool,
    thread: OnceLock<std::thread::Thread>,
}

struct Crew {
    jobs: Mutex<VecDeque<Job>>,
    /// How many jobs `jobs` holds, changed under its lock: what a helper
    /// looks at while it waits for work, without the lock.
    queued: AtomicUsize,
    hands: Vec<Hand>,
}

/// Since the frame line last took them: jobs given to the crew, jobs a
/// helper ran, sleeping helpers woken, and microseconds the submitting
/// thread stood waiting for jobs of a group that a helper was still
/// running (after it had done what was left of them itself).
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Helpers whose threads were made, and of those the ones that have begun
/// to run.
static SPAWNED: AtomicUsize = AtomicUsize::new(0);
static ALIVE: AtomicUsize = AtomicUsize::new(0);

/// How many helpers: `VICTORY_CREW_THREADS`, or a quarter of the logical
/// processors, 1 to 5.
fn threads() -> usize {
    match std::env::var("VICTORY_CREW_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) => n.min(16),
        None => std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(1, 5),
    }
}

/// How long a helper looks for work after its last job before it sleeps
/// (`VICTORY_CREW_SPIN_US`, 0 to 20,000 microseconds; 500 by default).
fn spin() -> std::time::Duration {
    static US: OnceLock<u64> = OnceLock::new();
    std::time::Duration::from_micros(*US.get_or_init(|| {
        std::env::var("VICTORY_CREW_SPIN_US").ok().and_then(|v| v.trim().parse::<u64>().ok()).map_or(500, |us| us.min(20_000))
    }))
}

static CREW: OnceLock<Option<Crew>> = OnceLock::new();

/// The crew, its helpers started on the first call; `None` without helpers.
fn crew() -> Option<&'static Crew> {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let crew = CREW
        .get_or_init(|| {
            let n = threads();
            (n > 0).then(|| Crew {
                jobs: Mutex::new(VecDeque::new()),
                queued: AtomicUsize::new(0),
                hands: (0..n)
                    .map(|_| Hand {
                        sleeping: AtomicBool::new(false),
                        thread: OnceLock::new(),
                    })
                    .collect(),
            })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        for i in 0..crew.hands.len() {
            match std::thread::Builder::new().name(format!("victory-wide-crew-{i}")).spawn(move || helper(i)) {
                Ok(_) => {
                    SPAWNED.fetch_add(1, Ordering::AcqRel);
                }
                Err(e) => tracing::warn!(error = %e, "a helper of the crew could not be started"),
            }
        }
        tracing::info!(
            "the crew of helpers for the submitting thread's preparing: {} threads (VICTORY_CREW_THREADS), each looks for work {} us after its last job before it sleeps (VICTORY_CREW_SPIN_US)",
            SPAWNED.load(Ordering::Acquire),
            spin().as_micros()
        );
    });
    Some(crew)
}

/// Whether jobs given to the crew run beside the thread that gives them: a
/// helper's thread was made (it may not have begun to run yet: its jobs
/// wait for it, or for the thread that waits for them). Starts the crew.
pub(crate) fn ready() -> bool {
    crew().is_some() && SPAWNED.load(Ordering::Acquire) > 0
}

/// A helper's life: jobs while there are any, a look for more for
/// [`spin`], then sleep until a job wakes it.
fn helper(i: usize) {
    let Some(crew) = CREW.get().and_then(Option::as_ref) else {
        return;
    };
    let hand = &crew.hands[i];
    let _ = hand.thread.set(std::thread::current());
    crate::framelog::keep_on_fast_cores(&format!("victory-wide-crew-{i}"));
    ALIVE.fetch_add(1, Ordering::AcqRel);
    let spin = spin();
    let mut idle = std::time::Instant::now();
    loop {
        if let Some(job) = crew.pop() {
            COUNTS[1].fetch_add(1, Ordering::Relaxed);
            run(job);
            idle = std::time::Instant::now();
            continue;
        }
        if idle.elapsed() < spin {
            for _ in 0..32 {
                std::hint::spin_loop();
            }
            continue;
        }
        // Asleep from here, unless a job came meanwhile: who gives a job
        // after this store sees it and wakes this helper (both sides are
        // sequentially consistent: either the look below sees the job, or
        // the giver's look sees this helper asleep).
        hand.sleeping.store(true, Ordering::SeqCst);
        if crew.queued.load(Ordering::SeqCst) > 0 {
            hand.sleeping.store(false, Ordering::SeqCst);
            continue;
        }
        // (A wake given before this park makes it return at once; one with
        // no reason looks for work and sleeps again.)
        std::thread::park();
        hand.sleeping.store(false, Ordering::SeqCst);
        idle = std::time::Instant::now();
    }
}

impl Crew {
    /// The first job waiting, if any.
    fn pop(&self) -> Option<Job> {
        if self.queued.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let job = jobs.pop_front()?;
        self.queued.fetch_sub(1, Ordering::SeqCst);
        Some(job)
    }

    /// The first job waiting of the group `tally` counts, if any.
    fn pop_of(&self, tally: &Arc<Tally>) -> Option<Job> {
        if self.queued.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let at = jobs.iter().position(|job| Arc::ptr_eq(&job.tally, tally))?;
        let job = jobs.remove(at)?;
        self.queued.fetch_sub(1, Ordering::SeqCst);
        Some(job)
    }

    /// Queues `job` and wakes one sleeping helper, if one sleeps.
    fn push(&self, job: Job) {
        {
            let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
            jobs.push_back(job);
            self.queued.fetch_add(1, Ordering::SeqCst);
        }
        for hand in &self.hands {
            if hand.sleeping.load(Ordering::SeqCst) && hand.sleeping.swap(false, Ordering::SeqCst) {
                if let Some(thread) = hand.thread.get() {
                    COUNTS[2].fetch_add(1, Ordering::Relaxed);
                    thread.unpark();
                }
                return;
            }
        }
    }
}

/// Runs a job and counts it done; a panic is kept for the group's waiter.
fn run(job: Job) {
    let Job { tally, work } = job;
    if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
        let mut first = tally.panic.lock().unwrap_or_else(|e| e.into_inner());
        if first.is_none() {
            *first = Some(panic);
        }
    }
    tally.left.fetch_sub(1, Ordering::Release);
}

/// Jobs given to the crew together, waited for together ([`Group::wait`]).
/// A group let go without the wait leaves its jobs to run: they own what
/// they work on.
pub(crate) struct Group {
    tally: Arc<Tally>,
}

/// A new group of jobs.
pub(crate) fn group() -> Group {
    Group {
        tally: Arc::new(Tally::default()),
    }
}

impl Group {
    /// Gives `work` to the crew; done at once on this thread if the crew
    /// has no helper.
    pub(crate) fn spawn(&self, work: impl FnOnce() + Send + 'static) {
        self.tally.left.fetch_add(1, Ordering::AcqRel);
        let job = Job {
            tally: self.tally.clone(),
            work: Box::new(work),
        };
        COUNTS[0].fetch_add(1, Ordering::Relaxed);
        match crew() {
            Some(crew) if SPAWNED.load(Ordering::Acquire) > 0 => crew.push(job),
            _ => run(job),
        }
    }

    /// Whether every job of the group is done.
    pub(crate) fn done(&self) -> bool {
        self.tally.left.load(Ordering::Acquire) == 0
    }

    /// Returns when every job of the group is done: the ones no helper
    /// has taken yet are done here, then this thread waits for those that
    /// helpers are running (a piece of work is a few hundred microseconds
    /// at most). A panic of a job goes on here.
    pub(crate) fn wait(self) {
        if let Some(crew) = CREW.get().and_then(Option::as_ref) {
            while let Some(job) = crew.pop_of(&self.tally) {
                run(job);
            }
        }
        if !self.done() {
            let started = std::time::Instant::now();
            let mut looks = 0u32;
            while !self.done() {
                if looks < 4096 {
                    std::hint::spin_loop();
                    looks += 1;
                } else {
                    std::thread::yield_now();
                }
            }
            COUNTS[3].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let panic = self.tally.panic.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(panic) = panic {
            std::panic::resume_unwind(panic);
        }
    }
}

/// With the frame line: what the crew did per frame over `frames`. Nothing
/// while it has never been given a job.
pub(crate) fn line(frames: f64) {
    if CREW.get().is_none() {
        return;
    }
    let [given, helped, woken, stood] = [0, 1, 2, 3].map(|i| COUNTS[i].swap(0, Ordering::Relaxed) as f64 / frames);
    tracing::info!(
        "the crew of helpers per frame ({} threads, VICTORY_CREW_THREADS): {given:.0} jobs given, {helped:.0} of them run by a helper (the rest by the thread that waited for them), {woken:.0} sleeping helpers woken; the submitting thread stood {:.2} ms waiting for jobs a helper was still running",
        ALIVE.load(Ordering::Relaxed),
        stood / 1e3
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every job of a group runs once, and the wait returns only when all
    /// of them are done, whichever threads ran them; groups given at once
    /// keep their own counts.
    #[test]
    fn a_group_waits_for_all_its_jobs_and_no_other() {
        let ran: Arc<Vec<AtomicUsize>> = Arc::new((0..400).map(|_| AtomicUsize::new(0)).collect());
        let (a, b) = (group(), group());
        for i in 0..400 {
            let ran = ran.clone();
            let work = move || {
                // A little work, so that helpers take some of them.
                let mut x = i as u64;
                for _ in 0..2_000 {
                    x = std::hint::black_box(x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1));
                }
                let _ = x;
                ran[i].fetch_add(1, Ordering::SeqCst);
            };
            if i % 2 == 0 { a.spawn(work) } else { b.spawn(work) }
        }
        a.wait();
        for i in (0..400).step_by(2) {
            assert_eq!(ran[i].load(Ordering::SeqCst), 1, "job {i} of the first group");
        }
        b.wait();
        for (i, n) in ran.iter().enumerate() {
            assert_eq!(n.load(Ordering::SeqCst), 1, "job {i}");
        }
    }

    /// A group with nothing in it, or whose jobs are all done, is waited
    /// for at once; a panic in a job comes out of the wait.
    #[test]
    fn an_empty_group_and_a_panic() {
        group().wait();
        let g = group();
        g.spawn(|| {});
        g.wait();
        let g = group();
        g.spawn(|| panic!("a job's panic"));
        g.spawn(|| {});
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.wait()));
        assert!(caught.is_err());
    }

    /// Stages given one after another from one thread, as a submission's
    /// flushes are: each finds its jobs done when its wait returns, and a
    /// long job of another group (the heap pass) runs beside them.
    #[test]
    fn stages_one_after_another_beside_a_long_job() {
        let long = group();
        let finished = Arc::new(AtomicBool::new(false));
        {
            let finished = finished.clone();
            long.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(30));
                finished.store(true, Ordering::SeqCst);
            });
        }
        for stage in 0..50usize {
            let sum = Arc::new(AtomicUsize::new(0));
            let g = group();
            for k in 0..(stage % 7) {
                let sum = sum.clone();
                g.spawn(move || {
                    sum.fetch_add(k + 1, Ordering::SeqCst);
                });
            }
            g.wait();
            let n = stage % 7;
            assert_eq!(sum.load(Ordering::SeqCst), n * (n + 1) / 2, "stage {stage}");
        }
        long.wait();
        assert!(finished.load(Ordering::SeqCst));
    }
}
