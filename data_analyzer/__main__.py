import sys

from lib import *

PROGRAM_NAME = sys.argv[0]

def usage():
    print(f"{PROGRAM_NAME} <command> [args...]")
    print("    get_pokemons <outputfile> <datasets...> salva nel file di output tutti i pokemon che trovi")
    print("    get_teams <outputfile> <dataset> <game id> <pokemon_file> prendi le squadre che trovi di un certo id")

def main():
    try:
        command = sys.argv[1]
    except IndexError:
        print("[ERROR] no command")
        usage()
        exit(1)
    try:
        database_path = sys.argv[2]
    except IndexError:
        print("[ERROR] no db specified")
        usage()
        exit(1)
    if command == "get_pokemons":
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            pokemons = get_pokemons(conn.cursor())
            print(pokemons)
    elif command == "get_teams":
        try:
            game_id = int(sys.argv[3])
            _set = sys.argv[4]
            if _set not in ["Test", "Train"]:
                print(f"[ERROR] the set {_set} doesn't exists")
                usage()
                exit(1)
        except IndexError:
            print("[ERROR] not enough arguments")
            usage()
            exit(1)
        except ValueError:
            print("[ERROR] the game id is not an integer")
            usage()
            exit(1)
        team1,team2 = None, None
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            team1, team2 = get_teams_complete(cur, game_id, _set)
            print(f"Team1:\n{team1}")
            print(f"Team2:\n{team2}")
            
            print(f"Avg Team1:\n{get_team_pokemon_avg_pd(team1)}")
            print(f"Avg Team2:\n{get_team_pokemon_avg_pd(team2)}")
    elif command == "avg":
        id_battle = int(sys.argv[3])
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            avg = get_team_pokemon_avg(conn.cursor(), id_battle)
            print(avg)
            all_avg = get_avg_pokemon(conn.cursor())
            print(all_avg)
    elif command == "pokemon_state":
        id_battle = int(sys.argv[3])
        _set = sys.argv[4]
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            team1, team2 = get_teams(cur, id_battle, _set)
            print(f"Team1:\n{team1}")
            print(f"Team2:\n{team2}")
            status1 = check_status_pokemon(cur, team1,True, id_battle)
            status2 = check_status_pokemon(cur, team2,False, id_battle)
            print(f"Status1:\n{status1}")
            print(f"Status2:\n{status2}")
            
    else:
        print(f"[ERROR] unknown command {command}")
        usage()
        exit(1)
    
main()