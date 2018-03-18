import os
import subprocess
import random

TRAIN_TEST_SPLIT = 80

FILE_NAMES = [
    "20021010_easy_ham.tar.bz2",
    # "20021010_hard_ham.tar.bz2",
    "20021010_spam.tar.bz2",
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    # "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
    "20050311_spam_2.tar.bz2"
]

LOCAL_DIR = "./data/"


def download():
    for file_name in FILE_NAMES:
        remote_url = "https://spamassassin.apache.org/old/publiccorpus/" + file_name
        local_url = LOCAL_DIR + file_name
        subprocess.run(["wget", remote_url, "--directory-prefix="+LOCAL_DIR])


def extract():
    for file_name in FILE_NAMES:
        local_url = LOCAL_DIR + file_name
        subprocess.run(["bunzip2", local_url])

        local_url = local_url.replace(".bz2", "")
        subprocess.run(["tar", "-xvf", local_url])

        subprocess.run(["rm", local_url])

    return [
        "easy_ham/",
        "easy_ham_2/",
        # "hard_ham/",
        "spam/",
        "spam_2/"
    ]


def split_train_test(dataset_directories):
    for directory in dataset_directories:
        files = [f for f in os.listdir(directory)]
        for f in files:
            train = random.randint(0, 100) > TRAIN_TEST_SPLIT
            if "ham" in directory:
                spam_or_ham = "ham"
            elif "spam" in directory:
                spam_or_ham = "spam"
            else:
                raise Exception("Unkown type spam or ham")
            local_path = directory + f

            if train:
                subprocess.run(["mv", local_path, "data/train/" + spam_or_ham + "/"])
            else:
                subprocess.run(["mv", local_path, "data/test/" + spam_or_ham + "/"])

        subprocess.run(["rm", "-r", directory])
        subprocess.run(["rm", "data/train/spam/cmds"])
        subprocess.run(["rm", "data/train/ham/cmds"])
        subprocess.run(["rm", "data/test/spam/cmds"])
        subprocess.run(["rm", "data/test/ham/cmds"])


download()
directories = extract()
split_train_test(directories)
