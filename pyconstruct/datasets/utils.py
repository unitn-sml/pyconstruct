
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
        'names': ['letter.data'],
        'checksums': ['ca5467fb4e87183cec0f8fabbf770b82']
    },
    'conll00': {
        'urls': [
            'https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz',
            'https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz'
        ],
        'names': ['train.txt', 'test.txt'],
        'checksums': [
            '2e2f24e90e20fcb910ab2251b5ed8cd0',
            '56944df34be553b72a2a634e539a0951'
        ]
    },
    'horseseg': {
        'urls': [
            'https://pub.ist.ac.at/~akolesnikov/HDSeg/HDSeg.tar',
            'https://pub.ist.ac.at/~akolesnikov/HDSeg/data.tar'
        ],
        'names': ['train.txt', 'test.txt'],
        'checksums': [
            '2e2f24e90e20fcb910ab2251b5ed8cd0',
            '56944df34be553b72a2a634e539a0951'
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


def fetch(source, base=None, verbose=True):
    """Fetches a dataset from the given source.

    Parameters
    ----------
    source : str
        The source key of the dataset to fetch.
    base : str
        The base directory where to save the dataset files.
    verbose : bool
        Wether to print progress output.

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

    names = []
    for i in range(len(source_info['urls'])):
        out_name = os.path.join(out_dir, source_info['names'][i])
        chk = source_info['checksums'][i]
        if not os.exist(outname) or checksum(out_name) != chk:
            url = source_info['urls'][i]
            if verbose:
                print('Downloading file: {}'.format(url))
            with urllib.request.urlopen(url) as response:
                with NamedTemporaryFile(
                    delete=False, dir=out_dir, mode='wb'
                ) as tmp:
                    copy(response, tmp, verbose=verbose)
                    tmp_name = tmp.name
            if zipfile.is_zipfile(tmp_name) or tarfile.is_tarfile(tmp_name):
                if verbose:
                    print('Extracting archive into: {}'.format(out_name))
                shutil.unpack_archive(tmp_name, out_dir)
                os.remove(tmp_name)
                names.append(out_name)
            elif is_gzip(tmp_name):
                if verbose:
                    print('Extracting archive into: {}'.format(out_name))
                unpack_gzip(tmp_name, out_name)
                os.remove(tmp_name)
                names.append(out_name)
            else:
                if verbose:
                    print('Moving file into: {}'.format(out_name))
                shutil.move(tmp_name, out_name)
                names.append(out_name)
    return names


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


def is_gzip(file_name):
    with open(file_name, 'rb') as f:
        return binascii.hexlify(f.read(2)) == b'1f8b'


def unpack_gzip(file_name, out_name):
    import gzip
    with gzip.open(file_name, 'rb') as tmp, open(out_name, 'w+') as out:
        shutil.copyfileobj(tmp, out)

