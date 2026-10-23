import unittest

from append_nested_spans import may_append


def span(start, end, hid='HP:0000001'):
    return {'offset': start, 'length': end-start, 'identifier': hid}


class NestedSpanPolicy(unittest.TestCase):
    def test_distinct_containment_both_directions_and_shared_edge(self):
        for candidate in [span(12, 18), span(10, 18), span(8, 22)]:
            self.assertTrue(may_append(candidate, [span(10, 20, 'HP:0000002')]))

    def test_identity_crossing_and_nonoverlap_rejected(self):
        prior = [span(10, 20, 'HP:0000002')]
        for candidate in [span(10, 20), span(15, 25), span(20, 25),
                          span(12, 18, 'HP:0000002'),
                          span(12, 18, 'HP:0000001;HP:0000002')]:
            self.assertFalse(may_append(candidate, prior))

    def test_every_collision_checked_including_newly_added(self):
        candidate = span(10, 30)
        self.assertTrue(may_append(candidate, [span(12, 15, 'HP:0000002'),
                                              span(20, 25, 'HP:0000003')]))
        self.assertFalse(may_append(candidate, [span(12, 15, 'HP:0000002'),
                                               span(25, 35, 'HP:0000003')]))


if __name__ == '__main__':
    unittest.main()
