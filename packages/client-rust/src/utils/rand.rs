use rand::seq::IndexedRandom;
use std::cell::RefCell;

thread_local! {
    static RNG: RefCell<rand::rngs::ThreadRng> = RefCell::new(rand::rng());
}

pub fn pick_one<T>(items: &[T]) -> &T {
    RNG.with(|rng| items.choose(&mut *rng.borrow_mut()).unwrap())
}
