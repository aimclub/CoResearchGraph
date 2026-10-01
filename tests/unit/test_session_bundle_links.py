"""An imported bundle's delivered links point at the new session."""
from CoScientist.web.session_bundle import _session_link_rebaser

OLD = "/api/users/user_old/sessions/session_old/artifacts/fig.png"
NEW = "/api/users/user_new/sessions/session_new/artifacts/fig.png"


def test_rebases_relative_and_absolute_links_in_nested_data():
    rebase = _session_link_rebaser("user_old", "session_old", "user_new", "session_new")
    data = [{"content": f"![fig]({OLD}) and http://127.0.0.1:8000{OLD}", "n": 3}]

    assert rebase(data) == [
        {"content": f"![fig]({NEW}) and http://127.0.0.1:8000{NEW}", "n": 3}
    ]


def test_leaves_other_sessions_and_missing_origin_alone():
    other = "/api/users/user_x/sessions/session_old/artifacts/fig.png"
    rebase = _session_link_rebaser("user_old", "session_old", "user_new", "session_new")
    assert rebase(other) == other

    untouched = _session_link_rebaser(None, None, "user_new", "session_new")
    assert untouched({"a": OLD}) == {"a": OLD}
