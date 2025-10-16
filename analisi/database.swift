Team(_id: serial_)

Pokemon(_name: String _, base_hp: IntegerGEZ, base_atk: IntegerGEZ, base_def: IntegerGEZ, base_spa: IntegerGEZ, base_spd: IntegerGEZ, base_spe: IntegerGEZ)

Level(_id: serial_, pokemon: String, team: Integer, level: IntegerGEZ)
    FK: pokemon references Pokemon(name)
    FK: team references Team(id)

Battle(_id: serial_,id_battle: IntegerGEZ , *result: bool, player: IntegerGEZ, p2_lead_pokemon: String, p2_pokeon_level: IntegerGEZ)
    FK: team references Team(id)
    KF: p2_lead_pokemon references Pokemon(name)

Dataset(_id: serial_, type: DatasetType)

bat_dat(_dataset: Integer, battle: IntegerGEZ _)
    FK: dataset references Dataset(id)
    FK: battle references Battle(id)

MoveType(_name: String _)

MoveCategory(_name: String _)

Move(_name: String, pokemon: String _, base_power: IntegerGEZ, accuracy: IntegerGEZ, priority: IntegerGEZ,type: String, category: String)
    FK: pokemon references Pokemon(name)
    FK: type references MoveType(name)
    FK: category references MoveCategory(name)

PokemonState(_id: serial_, hp_pct: IntegerGEZ, boost_atk: IntegerGEZ, boost_def: IntegerGEZ, boost_spa: IntegerGEZ, boost_spd: IntegerGEZ, boost_spe: IntegerGEZ, pokemon: String, *move: String)
    FK: pokemon references Pokemon(name)
    FK: move references Move(name)

Turn(_id: IntegerGEZ, battle: IntegerGEZ_, p1_state: Integer, p2_state: Integer)
    FK: battle references Battle(id)
    FK: p1_state references PokemonState(id)
    FK: p2_state references PokemonState(id)
    constraint: p1_state != p2_state

Effect(_name: String _)

eff_pok(_effect: String, state: Integer _)
    FK: effect references Effect(name)
    FK: state references PokemonState(id)