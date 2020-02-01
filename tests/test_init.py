from tinydb import Query, TinyDB

# from nonlocal_smms.core import scattering_matrix


def test_database_import():
    db = TinyDB("src/nonlocal_smms/materials.json")
    assert db.all() != []


def test_database_query():
    db = TinyDB("src/nonlocal_smms/materials.json")
    query = Query()

    assert db.search(query.material == "vacuum")[0] == {
        "beta_c": 0,
        "beta_l": 0,
        "beta_t": 0,
        "rho": 1,
        "eps_inf_pe": 1,
        "eps_inf_pa": 1,
        "eps_0_pe": 1,
        "eps_0_pa": 1,
        "wto_pe": 0,
        "wto_pa": 0,
        "gamma": 0,
        "wlo_pa": 0.0,
        "wlo_pe": 0.0,
        "material": "vacuum",
    }
