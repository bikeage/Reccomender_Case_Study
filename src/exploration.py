from bs4 import BeautifulSoup
import pandas as pd
import re

def get_ratings_and_jokes(ratings_filepath='data/ratings.dat', jokes_filepath='data/jokes.dat'):
    '''
    filepath likely this:  '../data/ratings.dat'
    '''
    ratings_df = pd.read_csv(ratings_filepath, delimiter='\t')
    jokes = []
    joke_ids = []

    with open(jokes_filepath,'r') as f:
        exp = r'([0-9]+):$'
        matcher = re.compile(exp)
        current_joke_html = ''
        current_joke_id = None
        for l in f:
            match = matcher.match(l)
            if match:
                if current_joke_id is None:
                    current_joke_id = int(match.group(1))
                    continue

                joke_text = parse_joke(current_joke_html)
                jokes.append(joke_text.strip())
                joke_ids.append(current_joke_id)

                # Reset joke stuff
                current_joke_html = ''
                current_joke_id = int(match.group(1))
            else:
                current_joke_html += l
                # print current_joke_html

    jokes_df = pd.DataFrame({
        'joke_text': jokes,
        'joke_id': joke_ids
    })

    return ratings_df, jokes_df

def parse_joke(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.text
