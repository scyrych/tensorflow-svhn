import os
import sys
import tarfile
import urllib.request

last_percent_reported = None
data_file_urls = {
    'data/train.tar.gz': 'http://ufldl.stanford.edu/housenumbers/train.tar.gz',
    'data/test.tar.gz': 'http://ufldl.stanford.edu/housenumbers/test.tar.gz',
    'data/extra.tar.gz': 'http://ufldl.stanford.edu/housenumbers/extra.tar.gz',
}


def _download_progress_hook(count, block_size, total_size):
    global last_percent_reported
    percent = int(count * block_size * 100 / total_size)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download_and_extract():
    if not os.path.exists('data'):
        os.makedirs('data')

    maybe_download()
    maybe_extract()


def maybe_download():
    for filename, url in data_file_urls.items():
        if not os.path.isfile(filename):
            print("Attempting to download", filename)
            saved_file, _ = urllib.request.urlretrieve(url, filename, _download_progress_hook)
            print("\nDownload Complete!")
            _validate_file_integrity(filename, saved_file)
        else:
            print("{} already downloaded".format(filename))


def _validate_file_integrity(filename, saved_file):
    stat_info = os.stat(saved_file)
    if stat_info.st_size == _get_expected_bytes(filename):
        print("Found and verified", saved_file)
    else:
        raise Exception("Failed to verify " + filename)


def _get_expected_bytes(filename):
    filename = os.path.basename(filename)
    if filename == "train_32x32.mat":
        byte_size = 182040794
    elif filename == "test_32x32.mat":
        byte_size = 64275384
    elif filename == "extra_32x32.mat":
        byte_size = 1329278602
    elif filename == "test.tar.gz":
        byte_size = 276555967
    elif filename == "train.tar.gz":
        byte_size = 404141560
    elif filename == "extra.tar.gz":
        byte_size = 1955489752
    else:
        raise Exception("Invalid file name " + filename)
    return byte_size


def maybe_extract(force=False):
    """ Helper function for extracting tarball files
        """
    ls_data = [f for f in os.listdir("data") if 'tar.gz' in f]
    os.chdir("data")

    for filename in ls_data:
        # Drop the file extension
        root = filename.split('.')[0]

        # If file is already extracted - return
        if os.path.isdir(root) and not force:
            print('%s already present - Skipping extraction of %s.' % (root, filename))
            continue

        # If file is a tarball file - extract it
        if filename.endswith("tar.gz"):
            print("Extracting %s ..." % filename)
            tar = tarfile.open(filename, "r:gz")
            tar.extractall()
            tar.close()

    os.chdir(os.path.pardir)
