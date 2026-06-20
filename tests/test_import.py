def test_import():
    import tidal_recovery
    from tidal_recovery.models import TiDALNet
    assert tidal_recovery.__version__
    assert TiDALNet is not None
