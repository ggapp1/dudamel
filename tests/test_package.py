import dudamel


def test_version():
    assert isinstance(dudamel.__version__, str)
    assert dudamel.__version__ == "0.1.0"
