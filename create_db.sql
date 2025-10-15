create domain String as varchar(100);

create domain IntegerGEZ as integer
    check(value >= 0);

create type DatasetType as enum('Train','Test');

-- =====================================================
-- TABELLA TEAM
-- =====================================================
CREATE TABLE Team (
    id SERIAL PRIMARY KEY
);

-- =====================================================
-- TABELLA POKEMON
-- =====================================================
CREATE TABLE Pokemon (
    name String PRIMARY KEY,
    base_hp IntegerGEZ NOT NULL,
    base_atk IntegerGEZ NOT NULL,
    base_def IntegerGEZ NOT NULL,
    base_spa IntegerGEZ NOT NULL,
    base_spd IntegerGEZ NOT NULL,
    base_spe IntegerGEZ NOT NULL
);

-- =====================================================
-- TABELLA LEVEL
-- =====================================================
CREATE TABLE Level (
    id SERIAL PRIMARY KEY,
    pokemon String NOT NULL,
    team IntegerGEZ NOT NULL,
    level IntegerGEZ NOT NULL,
    FOREIGN KEY (pokemon) REFERENCES Pokemon(name),
    FOREIGN KEY (team) REFERENCES Team(id)
);

-- =====================================================
-- TABELLA BATTLE
-- =====================================================
CREATE TABLE Battle (
    id IntegerGEZ PRIMARY KEY,
    result BOOLEAN,
    player IntegerGEZ NOT NULL,
    p2_lead_pokemon String NOT NULL,
    p2_pokeon_level IntegerGEZ NOT NULL,
    team IntegerGEZ NOT NULL,
    FOREIGN KEY (team) REFERENCES Team(id),
    FOREIGN KEY (p2_lead_pokemon) REFERENCES Pokemon(name)
);

-- =====================================================
-- TABELLA DATASET
-- =====================================================
CREATE TABLE Dataset (
    id SERIAL PRIMARY KEY,
    type DatasetType NOT NULL
);

-- =====================================================
-- TABELLA bat_dat
-- =====================================================
CREATE TABLE bat_dat (
    dataset IntegerGEZ NOT NULL,
    battle IntegerGEZ NOT NULL,
    PRIMARY KEY (dataset, battle),
    FOREIGN KEY (dataset) REFERENCES Dataset(id),
    FOREIGN KEY (battle) REFERENCES Battle(id)
);

-- =====================================================
-- TABELLA MoveType
-- =====================================================
CREATE TABLE MoveType (
    name String PRIMARY KEY
);

-- =====================================================
-- TABELLA MoveCategory
-- =====================================================
CREATE TABLE MoveCategory (
    name String PRIMARY KEY
);

-- =====================================================
-- TABELLA Move
-- =====================================================
CREATE TABLE PokemonMove (
    name String NOT NULL,
    pokemon String NOT NULL,
    base_power IntegerGEZ NOT NULL,
    accuracy IntegerGEZ NOT NULL,
    priority IntegerGEZ NOT NULL,
    type String NOT NULL,
    category String NOT NULL,
    PRIMARY KEY (name, pokemon),
    FOREIGN KEY (pokemon) REFERENCES Pokemon(name),
    FOREIGN KEY (type) REFERENCES MoveType(name),
    FOREIGN KEY (category) REFERENCES MoveCategory(name)
);

-- =====================================================
-- TABELLA PokemonState
-- =====================================================
CREATE TABLE PokemonState (
    id SERIAL PRIMARY KEY,
    hp_pct IntegerGEZ NOT NULL,
    boost_atk IntegerGEZ NOT NULL,
    boost_def IntegerGEZ NOT NULL,
    boost_spa IntegerGEZ NOT NULL,
    boost_spd IntegerGEZ NOT NULL,
    boost_spe IntegerGEZ NOT NULL,
    pokemon String NOT NULL,
    pok_move String,
    FOREIGN KEY (pokemon) REFERENCES Pokemon(name)
    -- FOREIGN KEY (pok_move) REFERENCES PokemonMove(name)
);

-- =====================================================
-- TABELLA Turn
-- =====================================================
CREATE TABLE Turn (
    id IntegerGEZ NOT NULL,
    battle IntegerGEZ NOT NULL,
    p1_state IntegerGEZ NOT NULL,
    p2_state IntegerGEZ NOT NULL,
    PRIMARY KEY (battle, id),
    FOREIGN KEY (battle) REFERENCES Battle(id),
    FOREIGN KEY (p1_state) REFERENCES PokemonState(id),
    FOREIGN KEY (p2_state) REFERENCES PokemonState(id),
    CONSTRAINT chk_diff_states CHECK (p1_state <> p2_state)
);

-- =====================================================
-- TABELLA Effect
-- =====================================================
CREATE TABLE Effect (
    name String PRIMARY KEY
);

-- =====================================================
-- TABELLA eff_pok
-- =====================================================
CREATE TABLE eff_pok (
    effect String NOT NULL,
    pok_state IntegerGEZ NOT NULL,
    PRIMARY KEY (effect, pok_state),
    FOREIGN KEY (effect) REFERENCES Effect(name),
    FOREIGN KEY (pok_state) REFERENCES PokemonState(id)
);