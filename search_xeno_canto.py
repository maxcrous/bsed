"""
Xeno Canto Scraper. Based on https://github.com/davipatti/birdbrain
Searching for and concurrently downloads content from https://www.xeno-canto.org

The program is called with a query. This query is handled in the same way as a user's
query submitted to xeno-canto's search bar. For example, if a user wanted all files
pertaining to common blackbirds, the query would simply be 'common blackbird'.

Xeno-canto allows search-flags in the query, such as recording rating or taxonomy membership.
If a

birds form the genus 'columba' and the species 'palumbus', the search query
would be
Example use:
$ python3 search_xeno_canto.py --query "Common wood pigeon gen: columba sp: palumbus q : A"
"""

import os
import re
import requests
import argparse
from p_tqdm import p_map
from tqdm import tqdm
from glob import glob


class XenoCantoRecord:
    """A xeno-canto recording metadata class"""

    def __init__(self, response):
        """ Arguments:
                response (dict) """

        self.response = response
        self.id = response.pop('id', 'none')
        self.genus = response.pop('gen', 'none')
        self.species = response.pop('sp', 'none')
        self.quality = response.pop('q', 'none')
        self.type = response.pop('type', 'none').replace(' ', '-').replace(',', '')

        self.file_url = 'https:' + response['file']
        self.file_size = 0
        self.audio_format = ''
        self.content_type = ''
        self.download_file_name = ''

    def download_audio(self, directory):
        """ Downloads the audio file for a recording object.

            Arguments:

                directory (str): Directory to save file in.
        """
        self.response = requests.get(self.file_url)
        self.file_size = self.response.headers['Content-Length']
        self.content_type = self.response.headers['Content-Type']
        self.audio_format = self.content_type.split('/')[1]
        valid_response = self.response.status_code == 200

        self.download_file_name = '{}.{}.{}.{}.{}.{}'.format(self.genus,
                                                             self.species,
                                                             self.id,
                                                             self.quality,
                                                             self.type,
                                                             self.audio_format)

        self.download_file_name = self.download_file_name.replace('/', '-')

        path = os.path.join(directory, self.download_file_name)
        downloaded_already = os.path.exists(path)

        if valid_response and not downloaded_already:
            with open(path, 'wb') as output_file:
                for data in self.response.iter_content():
                    output_file.write(data)


def remove_species(query):
    """ Removes species from query if one is specified.
        This is useful because Xeno-canto queries cannot specify a species.
    
        Arguments:
             query (str): Xeno-canto query.
            
        Returns: 
            query (str): The query without the species included
            sp (str): The species that was removed from the query
    """
    regex = re.compile('sp:[a-z]+', re.IGNORECASE)
    match = regex.search(query)

    if match:
        group = match.group()
        query = query.replace(group, '').rstrip().lstrip()
        sp = group.replace('sp:', '')

    else:
        sp = ''

    return query, sp


def post_query(query, page_number):
    """ Posts a query to the xeno-canto API.

    Arguments:
        query (str): Xeno canto search query.
        page_number (int): Which page to download.

    Returns:
        response_json (dict): A dictionary containing all metadata of a recording.
    """
    xeno_query_url = 'http://www.xeno-canto.org/api/2/recordings'
    params_dict = {'query': query, 'page': page_number}

    response = requests.get(url=xeno_query_url,
                            params=params_dict)

    valid_response = response.status_code == 200

    if valid_response:
        response_json = response.json()
        return response_json

    else:
        response.raise_for_status()


def fetch_single(recording_metadata, species, directory):
    recording = XenoCantoRecord(recording_metadata)
    regex = '*.{}.*'.format(recording.id)
    regex = os.path.join(directory, regex)
    already_has_path = glob(regex)

    if recording.species != species or already_has_path:
        return

    recording.download_audio(directory=directory)


def scrape(query,
           directory,
           max_downloads,
           all_pages):
    """ Scrapes audio files that match the user query.

        Arguments:
            query (str): Xeno canto search query.
            directory (str): Directory to save audio files to.
            max_downloads (int): The maximum amount of downloads.
            all_pages (bool): Whether to download more than one response page.
        Returns:
            None: Does not return anything. Does write audio files
                  to a specified directory.
    """

    current_page = 1
    query, species = remove_species(query)
    response_dict = post_query(query, current_page)
    total_length = len(response_dict['recordings'])
    response_pages = [response_dict]

    if all_pages:
        while len(response_dict['recordings']) == 500:
            current_page += 1
            response_dict = post_query(query, current_page)
            total_length += len(response_dict['recordings'])
            response_pages.append(response_dict)

    print('A total of {} files have been found.'.format(total_length))
    print('Commencing downloads.')
    for response_dict in tqdm(response_pages):
        recordings_to_fetch = list(response_dict['recordings'])
        recordings_to_fetch = recordings_to_fetch[:max_downloads]
        species_list = [species]*len(recordings_to_fetch)
        directory_list = [directory]*len(recordings_to_fetch)

        p_map(fetch_single, recordings_to_fetch, species_list, directory_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-q',
                        '--query',
                        help='Query. See https://www.xeno-canto.org/help/search for details. '
                             'Example: "gen: columba sp: palumbus q > : C"',
                        dest='query')

    parser.add_argument('-n',
                        '--max_number_downloads',
                        help='Maximum number of files to download.',
                        type=int,
                        dest='max_n_downloads',
                        default=10000)

    parser.add_argument('-p',
                        '--directory',
                        help='Directory to save downloads.',
                        dest='directory',
                        default='downloads')

    parser.add_argument('-m',
                        '--all_pages',
                        help='If set, fetch all available api responses (not just the first 500).',
                        dest='all_pages',
                        action='store_true')

    args = parser.parse_args()

    scrape(query=args.query,
           directory=args.directory,
           max_downloads=args.max_n_downloads,
           all_pages=args.all_pages)

