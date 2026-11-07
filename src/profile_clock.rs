//! CPU clocks for opt-in search profiling, excluding time descheduled by the OS.
//!
//! macOS constants and signatures are from the installed SDK's `usr/include/_time.h`.
//! Its `usr/share/man/man3/clock_gettime.3` documents both clocks as CPU time
//! consumed in user and kernel mode, with zero indicating failure for the
//! nanosecond-returning Darwin extension.

#[cfg(target_os = "macos")]
mod platform {
    use core::ffi::c_int;

    const CLOCK_PROCESS_CPUTIME_ID: c_int = 12;
    const CLOCK_THREAD_CPUTIME_ID: c_int = 16;

    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock_id: c_int) -> u64;
    }

    #[inline]
    fn cpu_ns(clock_id: c_int) -> Option<u64> {
        // SAFETY: These IDs are defined by the macOS SDK, and this function
        // accepts no pointers or caller-owned memory.
        let value = unsafe { clock_gettime_nsec_np(clock_id) };
        (value != 0).then_some(value)
    }

    #[inline]
    pub fn thread_cpu_ns() -> Option<u64> {
        cpu_ns(CLOCK_THREAD_CPUTIME_ID)
    }

    #[inline]
    pub fn process_cpu_ns() -> Option<u64> {
        cpu_ns(CLOCK_PROCESS_CPUTIME_ID)
    }
}

/// Cumulative CPU nanoseconds consumed by the calling thread.
///
/// Returns `None` if the clock is unavailable or fails. Thread measurements
/// must begin and end on the same thread.
#[inline]
pub fn thread_cpu_ns() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        platform::thread_cpu_ns()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Cumulative CPU nanoseconds consumed by all threads in the process.
///
/// Unlike elapsed time, this can increase faster than wall time when several
/// threads execute in parallel. Returns `None` if unavailable or on failure.
#[inline]
pub fn process_cpu_ns() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        platform::process_cpu_ns()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}
