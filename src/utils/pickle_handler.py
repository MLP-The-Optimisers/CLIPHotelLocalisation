import pickle

# Method for saving Python data as pickle files
def save_object(obj: any, repo: str, file: str):
    # Path constant to save the object
    PATH = f'{repo}/{file}.pkl'

    # Save as a pickle file
    with open(PATH, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Method for loading pickles into python data
def load_object(repo: str, file: str):
    # Path constant to save the object
    PATH = f'{repo}/{file}.pkl'
    print("loading this pickle file: ", PATH)

    with open(PATH, 'rb') as f:
        print("opened pickle file will now load")
        return pickle.load(f)