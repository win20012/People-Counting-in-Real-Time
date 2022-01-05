
def bigger():
    try:
        smaller()
    except:
        print('bigger except')

def smaller():
    try:
        print('smaller executed')
        raise ValueError
    except:
        print('smaller except')
        raise ValueError

bigger()