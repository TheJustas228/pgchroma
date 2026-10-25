def test_harness_both_servers_answer(clients):
    real, ours = clients
    assert real.heartbeat() > 0 and ours.heartbeat() > 0
    assert ours.list_collections() == [] and real.list_collections() == []
