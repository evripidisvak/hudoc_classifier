import csv
import os
import re
import sqlite3
import zipfile
from collections import Counter
from typing import List, Tuple

import enchant
import nltk
import requests
from english_words import get_english_words_set
from nltk.corpus import names, wordnet, words
from unicodedata import normalize

DB_PATH = "echr_cases.sqlite"
COUNTRIES_FILE = "countryInfo.txt"
CITIES_ZIP_FILE = "cities1000.zip"
CITIES_FILE = "cities1000.txt"
COUNTRY_CITIES_LIST = "county_cities_list.csv"

# URLs for the Geonames files
COUNTRY_INFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"
ALL_COUNTRIES_URL = "https://download.geonames.org/export/dump/cities1000.zip"
LEXPREDICT_URL = "https://raw.githubusercontent.com/LexPredict/lexpredict-legal-dictionary/refs/heads/master/en/legal/"

PLACEHOLDER_NAME = "[NAME]"
PLACEHOLDER_NUM = "[NUM]"
PLACEHOLDER_DATE = "[DATE]"
PLACEHOLDER_PLACE = "[PLACE]"

CUSTOM_ALLOWED_WORDS = [
    "section",
    "affidavit",
    "appellant",
    "appellate",
    "arbitration",
    "arraignment",
    "bailiff",
    "benchmark",
    "brief",
    "certiorari",
    "complaint",
    "counsel",
    "declaration",
    "decree",
    "default",
    "defendant",
    "demurrer",
    "deposition",
    "discovery",
    "docket",
    "estoppel",
    "evidence",
    "execution",
    "garnishment",
    "indictment",
    "injunction",
    "judgment",
    "jurisdiction",
    "jury",
    "lien",
    "litigation",
    "mandamus",
    "mediation",
    "motion",
    "notice",
    "objection",
    "petition",
    "plaintiff",
    "pleading",
    "precedent",
    "prosecution",
    "rebuttal",
    "recusal",
    "remand",
    "remedy",
    "resolution",
    "service",
    "settlement",
    "standing",
    "statement",
    "stipulation",
    "subpoena",
    "summons",
    "testimony",
    "tort",
    "transcript",
    "trial",
    "venue",
    "verdict",
    "witness",
    "writ",
    "abandonment",
    "abatement",
    "accord",
    "acquittal",
    "actus",
    "agency",
    "assignment",
    "assumption",
    "battery",
    "breach",
    "causation",
    "condition",
    "conspiracy",
    "consideration",
    "damages",
    "defamation",
    "duress",
    "duty",
    "easement",
    "equity",
    "felony",
    "fraud",
    "immunity",
    "implied",
    "infringement",
    "intent",
    "liability",
    "license",
    "malice",
    "negligence",
    "nuisance",
    "obligation",
    "possession",
    "privilege",
    "property",
    "proximate",
    "statutory",
    "tenancy",
    "title",
    "warranty",
    "ab initio",
    "actus reus",
    "ad hoc",
    "ad litem",
    "amicus curiae",
    "bona fide",
    "certiorari",
    "de facto",
    "de jure",
    "de minimis",
    "de novo",
    "ex ante",
    "ex parte",
    "ex post",
    "ex post facto",
    "force majeure",
    "habeas corpus",
    "in camera",
    "in limine",
    "in personam",
    "in re",
    "in rem",
    "inter alia",
    "ipse dixit",
    "ipso facto",
    "locus standi",
    "mens rea",
    "nisi prius",
    "non est factum",
    "obiter dicta",
    "per curiam",
    "per se",
    "prima facie",
    "pro bono",
    "pro se",
    "quantum meruit",
    "quid pro quo",
    "res ipsa loquitur",
    "res judicata",
    "stare decisis",
    "sub judice",
    "sui generis",
    "ultra vires",
    "voir dire",
    "acceptance",
    "agreement",
    "bilateral",
    "binding",
    "clause",
    "condition",
    "consent",
    "consideration",
    "covenant",
    "delivery",
    "endorsement",
    "executed",
    "grantor",
    "guarantee",
    "indemnity",
    "instrument",
    "modification",
    "negotiable",
    "offer",
    "option",
    "party",
    "performance",
    "recital",
    "revocation",
    "termination",
    "unilateral",
    "void",
    "waiver",
    "amendment",
    "article",
    "bill",
    "clause",
    "congress",
    "constitution",
    "enactment",
    "legislation",
    "provision",
    "regulation",
    "statute",
    "statutory",
    "subsection",
    "aka",
    "cfr",
    "corp",
    "esq",
    "et",
    "etc",
    "govt",
    "inc",
    "llc",
    "llp",
    "ltd",
    "no",
    "pl",
    "stat",
    "usc",
    "v",
    "vs",
    "eg",
    "echr",
    "hudoc",
    "appearence",
    "authorises",
    "characterised",
    "facto",
    "gmbh",
    "offences",
    "instalments",
    "summarised",
    "appendix",
    "list",
    "article",
    "decides",
    "declares",
    "court",
    "applicant",
    "acting",
    "lex",
    "specialis",
    "done",
    "admissibility",
    "offentlig",
    "keskitysleirejä",
    "apulaisoikeusasiamies",
    "biträdande",
    "law",
    "joukkoviestimissä",
    "registrar",
    "more",
    "than",
    "applicant",
    "mutandis",
    "mutatis",
    "totalling",
    "usd",
    "eur",
    "euro",
    "unforeseeability",
    "unhr",
    "specialising",
    "proprietate",
    "whatsapp",
    "practising",
    "neighbourhoods",
    "centres",
    "case",
    "cancelling",
    "interferences",
    "art",
    "reposted",
    "newsfeed",
    "nationalised",
    "law",
]
CUSTOM_DISALLOWED_WORDS = []
CUSTOM_DISALLOWED_PLACE_WORDS = [
        "Afghan",
        "Albanian",
        "Algerian",
        "Andorran",
        "Angolan",
        "Argentine",
        "Argentinian",
        "Armenian",
        "Australian",
        "Austrian",
        "Azerbaijani",
        "Bahamian",
        "Bahraini",
        "Bangladeshi",
        "Barbadian",
        "Bajan",
        "Belarusian",
        "Belgian",
        "Belizean",
        "Beninese",
        "Bhutanese",
        "Bolivian",
        "Bosnian",
        "Herzegovinian",
        "Batswana",
        "Botswanan",
        "Brazilian",
        "Bruneian",
        "Bulgarian",
        "Burkinabé",
        "Burundian",
        "Cambodian",
        "Cameroonian",
        "Canadian",
        "Cape Verdean",
        "Central African",
        "Chadian",
        "Chilean",
        "Chinese",
        "Colombian",
        "Comorian",
        "Congolese",
        "Congolese",
        "Costa Rican",
        "Croatian",
        "Cuban",
        "Cypriot",
        "Czech",
        "Danish",
        "Dane",
        "Djiboutian",
        "Dominican",
        "Dominican",
        "Ecuadorian",
        "Egyptian",
        "Salvadoran",
        "Salvadorean",
        "Equatoguinean",
        "Equatorial Guinean",
        "Eritrean",
        "Estonian",
        "Eswatini",
        "Swazi",
        "Ethiopian",
        "Fijian",
        "Finnish, Finn",
        "French",
        "Gabonese",
        "Gambian",
        "Georgian",
        "German",
        "Ghanaian",
        "Greek",
        "Hellenic",
        "Grenadian",
        "Guatemalan",
        "Guinean",
        "Bissau-Guinean",
        "Guyanese",
        "Haitian",
        "Honduran",
        "Hungarian",
        "Icelandic",
        "Icelander",
        "Indian",
        "Indonesian",
        "Iranian",
        "Persian",
        "Iraqi",
        "Irish",
        "Israeli",
        "Italian",
        "Ivorian",
        "Jamaican",
        "Japanese",
        "Jordanian",
        "Kazakh",
        "Kazakhstani",
        "Kenyan",
        "I-Kiribati",
        "Kuwaiti",
        "Kyrgyz",
        "Kyrgyzstani",
        "Kosovar",
        "Laotian",
        "Latvian",
        "Lebanese",
        "Basotho",
        "Lesothan",
        "Liberian",
        "Libyan",
        "Liechtensteiner",
        "Lithuanian",
        "Luxembourger",
        "Malagasy",
        "Malawian",
        "Malaysian",
        "Maldivian",
        "Malian",
        "Maltese",
        "Marshallese",
        "Mauritanian",
        "Mauritian",
        "Mexican",
        "Micronesian",
        "Moldovan",
        "Monégasque",
        "Monegasque",
        "Mongolian",
        "Montenegrin",
        "Moroccan",
        "Mozambican",
        "Burmese",
        "Namibian",
        "Nauruan",
        "Nepali",
        "Nepalese",
        "Dutch",
        "New Zealander",
        "Kiwi",
        "Nicaraguan",
        "Nigerien",
        "Nigerian",
        "North Korean",
        " Macedonian",
        "Norwegian",
        "Omani",
        "Pakistani",
        "Palauan",
        "Panamanian",
        "Papua New Guinean",
        "Paraguayan",
        "Peruvian",
        "Filipino",
        "Philippino",
        "Polish, Pole",
        "Portuguese",
        "Qatari",
        "Romanian",
        "Russian",
        "Rwandan",
        "Kittitian",
        "Nevisian",
        "Saint Lucian",
        "Vincentian",
        "Samoan",
        "Sammarinese",
        "Saudi",
        "Saudi Arabian",
        "Senegalese",
        "Serbian",
        "Seychellois",
        "Sierra Leonean",
        "Singaporean",
        "Slovak",
        "Slovenian",
        "Solomon Islander",
        "Somali",
        "South African",
        "South Korean",
        "South Sudanese",
        "Spanish",
        "Sri Lankan",
        "Sudanese",
        "Surinamese",
        "Swedish",
        "Swede",
        "Swiss",
        "Syrian",
        "Tajik",
        "Tajikistani",
        "Tanzanian",
        "Thai",
        "Togolese",
        "Tongan",
        "Tunisian",
        "Turkish",
        "Turkmen",
        "Tuvaluan",
        "Ugandan",
        "Ukrainian",
        "Emirati",
        "British",
        "American",
        "Uruguayan",
        "Uzbek",
        "Uzbekistani",
        "Ni-Vanuatu",
        "Vatican",
        "Holy See",
        "Venezuelan",
        "Vietnamese",
        "Yemeni",
        "Zambian",
        "Zimbabwean",
        ]
DATES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


class DatabaseManager:
    """Handles database operations"""

    def __init__(self, db_path: str):
        self.db_path = db_path

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Check if column exists
            cursor.execute("PRAGMA table_info(cases)")
            columns = [column[1] for column in cursor.fetchall()]

            if "dictionary_anonymized_judgement" not in columns:
                cursor.execute(
                    """
                        ALTER TABLE cases 
                        ADD COLUMN dictionary_anonymized_judgement TEXT
                    """
                )
                conn.commit()

            if "dictionary_removed_names" not in columns:
                cursor.execute(
                    """
                        ALTER TABLE cases 
                        ADD COLUMN dictionary_removed_names TEXT
                    """
                )
                conn.commit()

    def get_cases(self, limit: int = None) -> List[Tuple[int, str]]:
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT case_id, judgement FROM cases
                WHERE dictionary_anonymized_judgement IS NULL
            """

            if limit is not None:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query)
            return cursor.fetchall()

    def update_processed_case(self, case_id, anonymized_judgement, removed_words):
        """Update a single processed case in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                    UPDATE cases
                    SET dictionary_anonymized_judgement = ?,
                        dictionary_removed_names = ?
                    WHERE case_id = ?
                """,
                (
                    anonymized_judgement,
                    removed_words,
                    case_id,
                ),
            )
            conn.commit()


def download_lexpredict_dictionary():
    # Dictionary files we want to download
    dictionary_files = {
        "common_law": "common_law.csv",
    }

    legal_terms = set()

    print("Downloading LexPredict legal dictionary...")

    for category, filename in dictionary_files.items():
        try:
            if not os.path.isfile(filename):
                # Download the CSV file
                url = f"{LEXPREDICT_URL}/{filename}"
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for bad status codes

                with open(filename, mode="w", encoding="utf-8") as file:
                    file.write(response.text)

            with open(filename, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)

                # Extract terms from this category
                for row in reader:
                    term = row["Term"].lower().strip()
                    if len(term) > 0:
                        legal_terms.add(term)

        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")

    return legal_terms


def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {local_path}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded {local_path}.")
        else:
            raise Exception(f"Failed to download {url}: {response.status_code}")
    else:
        print(f"{local_path} already exists.")


def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        print(f"Extracted to {extract_to}.")
    else:
        print(f"{extract_to} already extracted.")


def parse_countries(file_path):
    print('Parsing countries')
    countries = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#"):  # Skip comments
                continue
            columns = line.strip().split("\t")
            if len(columns) > 4:  # Ensure valid line
                countries.append(columns[4])  # Column 5 contains the country name
    return countries


def parse_cities(file_path):
    print('Parsing cities')
    cities = set()
    forbidden_symbols = ['!', '£', '#', '_', '[', ']', '(', ')']

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) > 3:
                main_name = row[1]
                if any(char.isdigit() or char in main_name for char in forbidden_symbols):
                    continue
                # alternate_names = (
                #     row[3].split(",") if row[3] else []
                # )
                cities.add(main_name)
                # cities.update(alternate_names)
    return list(cities)


def combine_and_save():
    print(f'Creating combination list')
    countries = parse_countries(COUNTRIES_FILE)
    cities = parse_cities(CITIES_FILE)

    combined_list = sorted(set(countries + cities))

    with open('countries_list.csv', "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for place in countries:
            writer.writerow([place])
    print(f"Saved {len(countries)} places to countries_list.csv.")

    with open('cities_list.csv', "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for place in cities:
            writer.writerow([place])
    print(f"Saved {len(cities)} places to cities_list.csv.")

    with open(COUNTRY_CITIES_LIST, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for place in combined_list:
            writer.writerow([place])
    print(f"Saved {len(combined_list)} places to '{COUNTRY_CITIES_LIST}'.")


def load_csv_to_set(csv_file):
    print(f'Loading locations from {csv_file}')
    places_set = set()
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Skip empty rows
                places_set.add(row[0])
    print(f"Loaded {len(places_set)} places into a set.")
    return places_set


def get_countries_and_cities():
    if not os.path.exists(COUNTRY_CITIES_LIST):
        download_file(COUNTRY_INFO_URL, COUNTRIES_FILE)
        download_file(ALL_COUNTRIES_URL, CITIES_ZIP_FILE)
        extract_zip(CITIES_ZIP_FILE, CITIES_FILE)
        combine_and_save()

    places_set = load_csv_to_set(COUNTRY_CITIES_LIST)

    return places_set


def setup_word_lists():
    # Download required NLTK data
    nltk.download("words")
    nltk.download("wordnet")
    nltk.download("names")

    # Get words from NLTK's words corpus
    nltk_words = set(word.lower() for word in words.words())

    # Get words from WordNet
    wordnet_words = set(
        lemma.name().lower()
        for synset in wordnet.all_synsets()
        for lemma in synset.lemmas()
    )

    web2lowerset = get_english_words_set(["web2"], lower=True)

    legal_terms = download_lexpredict_dictionary()

    # Get words from PyEnchant's English dictionary
    enchant_dict = enchant.Dict("en_US")

    # Combine all sources into a master set
    english_words = (
        nltk_words.union(wordnet_words)
        .union(web2lowerset)
        .union(legal_terms)
        .union(CUSTOM_ALLOWED_WORDS)
    )

    # Get location words from WordNet
    countries_and_cities = get_countries_and_cities()
    location_words = set(normalize("NFC", place) for place in countries_and_cities)

    # Exclude names (including accented names)
    nltk_names = set(normalize("NFC", name) for name in names.words())
    force_remove_words = set(CUSTOM_DISALLOWED_WORDS + CUSTOM_DISALLOWED_PLACE_WORDS)

    # Remove names from english_words
    english_words = english_words - force_remove_words - nltk_names - location_words

    return english_words, force_remove_words, enchant_dict, nltk_names, location_words


def extract_words_from_text(text):
    """
    Extracts individual words from text, handling various edge cases.
    Returns a list of cleaned words.
    """
    # Convert to lowercase
    text = text.lower()

    # Replace hyphens with spaces to handle hyphenated words
    text = text.replace("-", " ")

    # Remove punctuation and split into words
    words = re.findall(r"\b[a-z']+\b", text)

    # Additional cleaning: remove standalone apostrophes and words that are just apostrophes
    words = [word.strip("'") for word in words if word.strip("'")]

    return words


def analyze_database_words(cases):
    # Set up our English word references
    english_word_set, force_remove_words, enchant_dict, known_names, known_locations = (
        setup_word_lists()
    )

    capitalized_custom_allowed = set(word.capitalize() for word in CUSTOM_ALLOWED_WORDS)
    known_names = known_names - capitalized_custom_allowed
    known_locations = known_locations - capitalized_custom_allowed

    print("Analyzing texts...")
    row_count = 0
    word_counter = Counter()

    for case_id, judgement in cases:
        if judgement:
            # Extract words from the text
            words = extract_words_from_text(judgement)
            # Update counter
            word_counter.update(words)

        row_count += 1
        if row_count % 100 == 0:
            print(f"Processed {row_count} rows")

    # Categorize words
    english_words = set()
    non_english_words = set()

    print("\nCategorizing words...")
    total_words = len(word_counter)
    processed = 0

    for word, count in word_counter.items():
        if (word.capitalize() in known_names) or (word.capitalize() in known_locations):
            is_english = False
            word = word.capitalize()
        else:
            word = word.lower()
            is_english = (
                word in english_word_set or enchant_dict.check(word)
            ) and word not in force_remove_words

        if is_english:
            english_words.add(word)
        else:
            non_english_words.add(word)

        processed += 1
        if processed % 1000 == 0:
            print(f"Categorized {processed}/{total_words} unique words")

    print(f"English words: {len(english_words)}")
    print(f"Non english words: {len(non_english_words)}")

    return english_words, non_english_words, known_names, known_locations


def clean_cases(cases, english_words, non_english_words, known_names, known_locations):
    db_manager = DatabaseManager(DB_PATH)
    counter = 1
    total_cases = len(cases)

    for case_id, judgement in cases:
        if not judgement:
            continue

        # Replace non-English words in the judgement
        replaced_words = []

        # Define a replacement function
        def replace_non_english(match):
            word = match.group(0)
            # Normalize the word to NFC Unicode form
            word_normalized = normalize("NFC", word)

            if word_normalized.lower() in CUSTOM_DISALLOWED_WORDS:
                return PLACEHOLDER_NAME

            if word_normalized.lower() in CUSTOM_DISALLOWED_PLACE_WORDS:
                return PLACEHOLDER_PLACE

            if word_normalized.lower() in CUSTOM_ALLOWED_WORDS:
                return word

            if word_normalized.isdigit():
                replaced_words.append(word)
                return PLACEHOLDER_NUM

            if word_normalized in DATES:
                replaced_words.append(word)
                return PLACEHOLDER_DATE

            if word_normalized in known_names:
                replaced_words.append(word)
                return PLACEHOLDER_NAME

            if word_normalized in known_locations:
                replaced_words.append(word)
                return PLACEHOLDER_PLACE

            if len(word_normalized) <= 2:
                return word

            parts = re.split(r"[-’']", word_normalized)

            for part in parts:
                # Preserve valid words with hyphens or apostrophes (e.g., case-law, Court’s)
                if re.match(r"^[a-zA-ZÀ-ÖØ-öø-ÿ’']+(?:-[a-zA-ZÀ-ÖØ-öø-ÿ’']+)*$", part):
                    # Check if the word is not English or legal
                    # if word in non_english_words or part not in english_words:
                    if (
                        part.lower() in non_english_words
                        or part.lower() not in english_words
                    ):
                        replaced_words.append(part)  # Add only non-English words
                        return PLACEHOLDER_NAME
                    return part

                # For all other cases, replace and add to replaced_words
                replaced_words.append(part)
                return PLACEHOLDER_NAME

        # Use regex to replace words in the text
        processed_judgement = re.sub(
            r"\b[\wÀ-ÖØ-öø-ÿ’'\-]+\b", replace_non_english, judgement, flags=re.UNICODE
        )

        # Update the database with the processed judgement and replaced words
        db_manager.update_processed_case(
            case_id,
            processed_judgement,
            ", ".join(replaced_words),  # Convert list of replaced words to a string
        )

        print(
            f"Processed case_id {case_id}: {len(replaced_words)} words replaced [{counter}/{total_cases} (Remaining {total_cases - counter})]"
        )
        counter += 1


if __name__ == "__main__":
    db_manager = DatabaseManager(DB_PATH)

    # Get unprocessed cases
    cases = db_manager.get_cases()
    total_cases = len(cases)
    print(f"Retrieved {total_cases} unprocessed cases from database")

    if total_cases > 0:
        english_words, non_english_words, known_names, known_locations = (
            analyze_database_words(cases)
        )
        clean_cases(
            cases, english_words, non_english_words, known_names, known_locations
        )
