from dynalloc_v2.experiment_windows import available_split_profiles, profile_payload, resolve_split_payload


def test_available_profiles_include_shifted_window():
    assert 'cv2000_final20y' in available_split_profiles()
    assert 'cv2006_final20y' in available_split_profiles()


def test_profile_payload_cv2006():
    payload = profile_payload('cv2006_final20y')
    assert payload['train_pool_end'] == '2005-12-31'
    assert payload['test_start'] == '2006-01-01'
    assert payload['end_date'] == '2025-12-31'


def test_resolve_split_payload_can_override_dates_on_top_of_profile():
    payload, meta = resolve_split_payload(
        base_profile='cv2000_final20y',
        split_profile_override='cv2006_final20y',
        train_start_override='1970-01-01',
        train_pool_end_override='2004-12-31',
        test_start_override='2005-01-01',
        end_date_override='2024-12-31',
    )
    assert payload['train_start'] == '1970-01-01'
    assert payload['train_pool_end'] == '2004-12-31'
    assert payload['test_start'] == '2005-01-01'
    assert payload['end_date'] == '2024-12-31'
    assert meta['split_profile'] == 'cv2006_final20y'
    assert meta['split_overrides']['test_start'] == '2005-01-01'
