#include <cassert>
#include <iostream>
#include <vector>

#include "minillm/minillm.h"

using namespace minillm;

void test_submit_and_admit() {
    std::cout << "  test_submit_and_admit...";
    PagedKVCache cache;
    auto st = cache.init(2, 4, 64, 16, 64);
    assert(st.ok());

    ContinuousBatchScheduler scheduler(cache, {.max_batch_size = 4});
    auto req0 = SequenceRequest{.sequence_id = 0, .prompt_tokens = {1, 2, 3}, .max_tokens = 10};
    st = scheduler.submit(req0);
    assert(st.ok());
    auto req1 = SequenceRequest{.sequence_id = 1, .prompt_tokens = {4, 5}, .max_tokens = 10};
    st = scheduler.submit(req1);
    assert(st.ok());
    assert(scheduler.waiting_count() == 2);

    st = scheduler.admit_waiting();
    assert(st.ok());
    assert(scheduler.active_count() == 2);
    assert(scheduler.waiting_count() == 0);
    assert(scheduler.has_active());
    assert(!scheduler.all_done());

    assert(scheduler.state(0).phase == SeqPhase::Prefilling);
    assert(scheduler.state(1).phase == SeqPhase::Prefilling);

    std::cout << " PASSED\n";
}

void test_max_batch_size() {
    std::cout << "  test_max_batch_size...";
    PagedKVCache cache;
    auto st = cache.init(2, 4, 64, 16, 64);
    assert(st.ok());

    ContinuousBatchScheduler scheduler(cache, {.max_batch_size = 2});
    auto req0 = SequenceRequest{.sequence_id = 0, .prompt_tokens = {1}, .max_tokens = 5};
    st = scheduler.submit(req0);
    assert(st.ok());
    auto req1 = SequenceRequest{.sequence_id = 1, .prompt_tokens = {2}, .max_tokens = 5};
    st = scheduler.submit(req1);
    assert(st.ok());
    auto req2 = SequenceRequest{.sequence_id = 2, .prompt_tokens = {3}, .max_tokens = 5};
    st = scheduler.submit(req2);
    assert(st.ok());

    st = scheduler.admit_waiting();
    assert(st.ok());

    assert(scheduler.active_count() == 2);
    assert(scheduler.waiting_count() == 1);

    std::cout << " PASSED\n";
}

void test_explicit_id_advances_auto_id() {
    std::cout << "  test_explicit_id_advances_auto_id...";
    PagedKVCache cache;
    auto st = cache.init(2, 4, 64, 16, 64);
    assert(st.ok());

    ContinuousBatchScheduler scheduler(cache, {.max_batch_size = 4});
    st = scheduler.submit({.sequence_id = 2, .prompt_tokens = {1}, .max_tokens = 4});
    assert(st.ok());
    st = scheduler.submit({.prompt_tokens = {2}, .max_tokens = 4});
    assert(st.ok());

    assert(scheduler.waiting_ids().size() == 2);
    assert(scheduler.waiting_ids()[0] == 2);
    assert(scheduler.waiting_ids()[1] == 3);

    st = scheduler.submit({.sequence_id = 2, .prompt_tokens = {3}, .max_tokens = 4});
    assert(!st.ok());

    std::cout << " PASSED\n";
}

void test_phase_transitions() {
    std::cout << "  test_phase_transitions...";
    PagedKVCache cache;
    auto st = cache.init(2, 4, 64, 16, 64);
    assert(st.ok());

    ContinuousBatchScheduler scheduler(cache, {.max_batch_size = 4});
    auto req = SequenceRequest{.sequence_id = 0, .prompt_tokens = {1, 2, 3, 4}, .max_tokens = 8};
    st = scheduler.submit(req);
    st = scheduler.admit_waiting();
    assert(st.ok());

    assert(scheduler.state(0).phase == SeqPhase::Prefilling);

    scheduler.mark_prefill_progress(0, 2);
    assert(scheduler.state(0).phase == SeqPhase::Prefilling);

    scheduler.mark_prefill_progress(0, 4);
    assert(scheduler.state(0).phase == SeqPhase::Decoding);

    scheduler.mark_token_generated(0, 42);
    assert(scheduler.state(0).generated_tokens.size() == 1);
    assert(scheduler.state(0).generated_tokens[0] == 42);

    for (int i = 0; i < 7; ++i) {
        scheduler.mark_token_generated(0, i);
    }
    assert(scheduler.state(0).phase == SeqPhase::Finished);

    std::cout << " PASSED\n";
}

void test_evict_finished() {
    std::cout << "  test_evict_finished...";
    PagedKVCache cache;
    auto st = cache.init(2, 4, 64, 16, 64);
    assert(st.ok());

    ContinuousBatchScheduler scheduler(cache, {.max_batch_size = 4});
    auto req0 = SequenceRequest{.sequence_id = 0, .prompt_tokens = {1}, .max_tokens = 2};
    st = scheduler.submit(req0);
    assert(st.ok());
    auto req1 = SequenceRequest{.sequence_id = 1, .prompt_tokens = {2}, .max_tokens = 20};
    st = scheduler.submit(req1);
    assert(st.ok());

    int free_before = cache.free_block_count();
    st = scheduler.admit_waiting();
    assert(st.ok());
    assert(scheduler.active_count() == 2);

    scheduler.mark_prefill_progress(0, 1);
    assert(scheduler.state(0).phase == SeqPhase::Decoding);

    scheduler.mark_token_generated(0, 10);
    scheduler.mark_token_generated(0, 11);
    assert(scheduler.state(0).phase == SeqPhase::Finished);

    scheduler.mark_prefill_progress(1, 1);
    assert(scheduler.state(1).phase == SeqPhase::Decoding);

    int free_mid = cache.free_block_count();
    assert(free_mid < free_before);

    scheduler.evict_finished();

    assert(scheduler.active_count() == 1);

    int free_after = cache.free_block_count();
    assert(free_after > free_mid);

    std::cout << " PASSED\n";
}

void test_collect_finished() {
    std::cout << "  test_collect_finished...";
    PagedKVCache cache;
    auto st = cache.init(2, 4, 64, 16, 64);
    assert(st.ok());

    ContinuousBatchScheduler scheduler(cache, {.max_batch_size = 4});
    auto req = SequenceRequest{.sequence_id = 0, .prompt_tokens = {1, 2}, .max_tokens = 3};
    st = scheduler.submit(req);
    st = scheduler.admit_waiting();
    assert(st.ok());

    scheduler.mark_prefill_progress(0, 2);
    assert(scheduler.state(0).phase == SeqPhase::Decoding);

    scheduler.mark_token_generated(0, 100);
    scheduler.mark_token_generated(0, 200);
    scheduler.mark_token_generated(0, 300);
    assert(scheduler.state(0).phase == SeqPhase::Finished);

    auto finished = scheduler.collect_finished();
    assert(finished.size() == 1);
    const auto expected_gen = std::vector<int32_t>{100, 200, 300};
    const auto expected_prompt = std::vector<int32_t>{1, 2};
    assert(finished[0].sequence_id == 0);
    assert(finished[0].prompt_tokens == expected_prompt);
    assert(finished[0].generated_tokens == expected_gen);
    assert(finished[0].finished);

    assert(!scheduler.all_done());

    scheduler.evict_finished();
    assert(scheduler.all_done());

    std::cout << " PASSED\n";
}

void test_all_done() {
    std::cout << "  test_all_done...";
    PagedKVCache cache;
    auto st = cache.init(1, 2, 64, 16, 32);
    assert(st.ok());

    ContinuousBatchScheduler scheduler(cache, {.max_batch_size = 2});
    assert(scheduler.all_done());

    auto req = SequenceRequest{.sequence_id = 0, .prompt_tokens = {1}, .max_tokens = 1};
    st = scheduler.submit(req);
    assert(st.ok());
    assert(!scheduler.all_done());
    assert(scheduler.has_waiting());

    st = scheduler.admit_waiting();
    assert(st.ok());
    assert(!scheduler.all_done());
    assert(scheduler.has_active());

    scheduler.mark_prefill_progress(0, 1);
    scheduler.mark_token_generated(0, 42);
    assert(scheduler.state(0).phase == SeqPhase::Finished);

    scheduler.evict_finished();
    assert(scheduler.all_done());

    std::cout << " PASSED\n";
}

int main() {
    std::cout << "ContinuousBatchScheduler tests:\n";
    test_submit_and_admit();
    test_max_batch_size();
    test_explicit_id_advances_auto_id();
    test_phase_transitions();
    test_evict_finished();
    test_collect_finished();
    test_all_done();
    std::cout << "All ContinuousBatchScheduler tests passed!\n";
    return 0;
}
