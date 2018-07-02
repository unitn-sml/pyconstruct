
import os
import shutil
import appdirs
import hashlib
import zipfile
import tarfile
import binascii
import urllib.request

from tempfile import NamedTemporaryFile


SOURCES = {
    'ocr': {
        'urls': ['http://ai.stanford.edu/~btaskar/ocr/letter.data.gz'],
        'raw_names': ['letter.data.gz'],
        'names': ['letter.data'],
        'checksums': ['ca5467fb4e87183cec0f8fabbf770b82'],
        'steps': [('download', 0), ('unpack', 0)]
    },
    'conll00': {
        'urls': [
            'https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz',
            'https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz'
        ],
        'raw_names': ['train.txt.gz', 'test.txt.gz'],
        'names': ['train.txt', 'test.txt'],
        'checksums': [
            '2e2f24e90e20fcb910ab2251b5ed8cd0',
            '56944df34be553b72a2a634e539a0951'
        ],
        'steps': [
            ('download', 0), ('unpack', 0), ('download', 1), ('unpack', 1)
        ]
    },
    'horseseg': {
        'urls': [
            'https://pub.ist.ac.at/~akolesnikov/HDSeg/HDSeg.tar',
            'https://pub.ist.ac.at/~akolesnikov/HDSeg/data.tar'
        ],
        'raw_names': ['HDSeg.tar', 'data.tar'],
        'names': ['train.txt', 'test.txt'],
        'checksums': [
            '2e2f24e90e20fcb910ab2251b5ed8cd0',
            '56944df34be553b72a2a634e539a0951'
        ],
        'steps': [('download', 0), ('unzip', 0), ('download', 1), ('unzip', 1)]
    },
    'equations': {
        'urls': [
            'http://sml.disi.unitn.it/software/equations/equations.tar.gz'
        ],
        'raw_names': ['equations.tar.gz'],
        'names': ['equations.pickle'],
        'checksums': [
            'b3b1bc045622950e207a791be06f21b1',
        ],
        'steps': [
            ('download', 0), ('unpack', 0)
        ]
    }
}


def data_dir(source, base=None):
    if base is None:
        base = appdirs.user_data_dir('pyconstruct')
    return os.path.join(base, 'data', source)


def get_paths(source, base=None):
    source_dir = data_dir(source, base)
    source_info = SOURCES[source]
    return [os.path.join(source_dir, name) for name in source_info['names']]


def exist(paths):
    for path in paths:
        if not os.path.exists(path):
            return False
    return True


def fetch(source, base=None, verbose=True, remove_raw=False):
    """Fetches a dataset from the given source.

    Parameters
    ----------
    source : str
        The source key of the dataset to fetch.
    base : str
        The base directory where to save the dataset files.
    verbose : bool
        Wether to print progress output.
    remove_raw : bool
        Wether to remove the download raw files.

    Returns
    -------
    files : [str]
        The list of files fetched.
    """
    if source not in SOURCES:
        raise ValueError('Invalid source.')

    out_dir = data_dir(source, base)
    os.makedirs(out_dir, exist_ok=True)

    source_info = SOURCES[source]

    for step, idx in source_info['steps']:
        {
            'download': download, 'unpack': unpack,
        }[step](source_info, idx, out_dir, verbose)

    if remove_raw:
        for raw_name in source_info['raw_names']:
            os.remove(os.path.join(out_dir, raw_name))


def download(source_info, idx, out_dir, verbose=True):
    raw_name = os.path.join(out_dir, source_info['raw_names'][idx])
    chk = source_info['checksums'][idx]
    if not os.path.isfile(raw_name) or checksum(raw_name) != chk:
        url = source_info['urls'][idx]
        if verbose:
            print('Downloading file: {}'.format(url))
        with urllib.request.urlopen(url) as response:
            with open(raw_name, mode='wb') as f:
                copy(response, f, verbose=verbose)
    elif verbose:
        print('Using previously downloaded source: {}.'.format(raw_name))


def copy(response, output_file, chunk_size=8192, verbose=False):
    total_size = int(response.info().get('Content-Length').strip())
    size = 0

    try:
        while True:
            chunk = response.read(chunk_size)
            size += len(chunk)

            output_file.write(chunk)

            if not chunk:
                break

            if verbose:
                percent = round((float(size) / total_size) * 100, 2)
                print('Downloaded {:>8d} of {:>8d} bytes ({:1>0.2f}%)'.format(
                    size, total_size, percent
                ), end='\r')
    except Exception:
        if verbose:
            print('\nDownload failed.', end='\n\n')
        raise
    if verbose:
        print('\nDownload complete.', end='\n\n')


def checksum(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def unpack(source_info, idx, out_dir, verbose):
    raw_name = os.path.join(out_dir, source_info['raw_names'][idx])
    out_name = os.path.join(out_dir, source_info['names'][idx])
    if verbose:
        print('Extracting archive: {}'.format(raw_name))
    if zipfile.is_zipfile(raw_name) or tarfile.is_tarfile(raw_name):
        shutil.unpack_archive(raw_name, out_dir)
    elif is_gzip(raw_name):
        unpack_gzip(tmp_name, out_name)
    else:
        raise ValueError('Archive type not recognized: {}.'.format(raw_name))


def is_gzip(file_name):
    with open(file_name, 'rb') as f:
        return binascii.hexlify(f.read(2)) == b'1f8b'


def unpack_gzip(file_name, out_name):
    import gzip
    with gzip.open(file_name, 'rt') as tmp, open(out_name, 'w+') as out:
        shutil.copyfileobj(tmp, out)

