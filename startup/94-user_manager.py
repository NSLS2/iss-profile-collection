
print(ttime.ctime() + ' >>>> ' + __file__)

class UserManager(PersistentListInteractingWithGUI):

    def __init__(self, json_file_path = f'{ROOT_PATH_SHARED}/settings/user_management/user_manager.json'):
        super().__init__(json_file_path)
        _, current_user = self.current_user()
        if current_user is not None:
            self._init_managers(current_user['sample_manager'], current_user['scan_manager'])



    # Class specific decorators
    @property
    def users(self):
        return self.items


    def find_user_index(self, first_name, last_name):
        for i, user in enumerate(self.users):
            if (user['first_name'] == first_name) and (user['last_name'] == last_name):
                return i
        return -1

    def find_user(self, first_name, last_name):
        idx = self.find_user_index(first_name, last_name)
        if idx == -1:
            return None
        return self.users[idx]

    @emit_list_update_signal_decorator
    def set_user(self, first_name, last_name, affiliation, email):
        index = self.find_user_index(first_name, last_name)
        if index == -1:
            uid = str(uuid.uuid4())
            sample_manager_filename = f'sample_manager_{str(uuid.uuid4())[0:7]}.json'
            scan_manager_filename = f'scan_manager_{str(uuid.uuid4())[0:7]}.json'
            user_dict =     {'uid': uid,
                            'first_name': first_name,
                            'last_name': last_name,
                            'affiliation': affiliation,
                            'email':email,
                            'sample_manager': sample_manager_filename,
                            'scan_manager': scan_manager_filename,
                            }

            self.users.append(user_dict)
        else:
            sample_manager_filename = self.users[index]['sample_manager']
            scan_manager_filename = self.users[index]['scan_manager']


        self._init_managers(sample_manager_filename, scan_manager_filename)
        johann_emission.initialized = False
        scan_sequence_manager.reset()
        batch_manager.reset()

        RE.md['PI'] = f'{first_name} {last_name}'
        RE.md['affiliation'] = affiliation
        RE.md['email'] = email

    def _init_managers(self, sample_manager_filename, scan_manager_filename):
        sample_manager.init_from_new_file(f'{ROOT_PATH_SHARED}/settings/user_management/{sample_manager_filename}')
        scan_manager.load_local_manager(f'{ROOT_PATH_SHARED}/settings/user_management/{scan_manager_filename}')

    def current_user(self):
        current_user_name = RE.md['PI'].split(' ')
        for indx, user in enumerate(self.users):
            if (user['first_name'] == current_user_name[0]) and (user['last_name'] == current_user_name[1]):
                return indx, user
        return None, None

    @emit_list_update_signal_decorator
    def add_run(self, proposal, saf, experimenters):
        _current_index, _current_user  = self.current_user()
        run={}
        run['start'] = ttime.ctime()
        run['timestamp'] = ttime.time()
        run['proposal'] = proposal
        run['saf'] = saf
        run['experimenters'] = experimenters
        if 'runs' in _current_user.keys():
            self.users[_current_index]['runs'].append(run)
        else:
            self.users[_current_index]['runs']=[run]
        RE.md['proposal'] = str(proposal)
        RE.md['saf'] = str(saf)
        RE.md['experimenters'] = experimenters

    @emit_list_update_signal_decorator
    def add_metadata_key(self, key):
        _current_index, _current_user = self.current_user()
        if 'metadata' in _current_user.keys():
            self.users[_current_index]['metadata'].append(key)
        else:
            self.users[_current_index]['metadata'] = [key]

    @emit_list_update_signal_decorator
    def remove_metadata_key(self, key):
        _current_index, _current_user = self.current_user()
        if 'metadata' in _current_user.keys():
            for _indx, _key in enumerate(self.users[_current_index]['metadata']):
                if _key == key:
                    self.users[_current_index]['metadata'].pop(_indx)

        else:
            self.users[_current_index]['metadata'] = [key]


user_manager = UserManager()



def quac():
    proposal = RE.md['proposal']
    import requests
    %cd ~
    headers = {'accept': 'application/json', }
    proposal_info = requests.get(f'https://api.nsls2.bnl.gov/v1/proposal/{proposal}', headers=headers).json()

    user_names = []
    for user in proposal_info['proposal']['users']:
        user_names.append(f"{user['last_name']}, {user['first_name']}")
    name_list = '" "'.join(user_names)
    os.system(f'./search_users.sh -- "{name_list}" > output.txt')

    file_path = 'output.txt'
    with open(file_path, 'r') as file:
        file_content = file.read()
    strings = file_content.split('\n')
    logins = []
    for string in strings:
        if (string.find(',') > -1) and (string.find('Running command') == -1):
            inds = [index for index, char in enumerate(string) if char == '|']
            logins.append(string[(inds[1] + 1):inds[2]].strip())

    print(f'The users for the experiment under are{proposal}')
    for name in logins:
        print(f'    {name}')








