""" TensorMONK's :: data :: LMDB """

import os
import io
import lmdb
import numpy as np
from PIL import Image as ImPIL
import msgpack
import base64
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class LMDB(object):
    r""" Creates and read a lmdb database.

    Creating a database:
    -------------------
        - Accepts str/int/float/np.ndarray values
        - When str endswith .png/.jpg/.jpeg/.tiff/.bmp (image path), reads the
        image in bytearray and saves to the database (full image path is also
        saved to the database - set show_image_name to True to output the image
        name during read).

    Reading a database:
    ------------------
        - Requires 0 <= idx < self.__len__()
        - All the str (excluding the once that end with IMAGE_TYPES)/int/float/
        np.ndarray retain their type and shape
        - str's ending with IMAGE_TYPES will return a pillow image. When
        show_image_name is True, (pillow image, image name) is returned.

    Example:
        >>> database = LMDB(file_name="./test.lmdb",
                           attributes=["image", "label"],
                           map_size=1024*1024)
        >>> database.start(write=True)
        >>> database.write((np.random.randn(100, 100), 4))
        >>> database.write((np.random.randn(100, 100), 6))
        >>> database.write((np.random.randn(100, 100), 4))
        >>> database.write((np.random.randn(100, 100), 6))
        >>> print(len(database))
        >>> database.stop()

        >>> database.start(write=False)
        >>> database.read(0)
        >>> database.read(1)
        >>> database.read(2)
        >>> database.read(3)
        >>> database.stop()

    Args:
        file_name (str): lmdb file name (full path)
        attributes (list/tuple): list/tuple of attributes within a sample
            Ex: ("image", "label"), ("image", "mask")
        map_size (int): size of database
        show_image_name (bool): return the image name for read()
        encrypt (bool): Will encrypt all the images and their names with a
            random key (saved in key_file_name)
        key_file_name (str): Required when encrypt=True, store the random
            encryption key for a new database or loaded the key to decrypt
            images

    ** No Guarantees or Warranties
    Few to note:
        - If you are not sure about encrypt, don't use it.
        - Do not save/send ".key" along with database file.
        - Obviously, using encrypt, will slow down the read and write process.
    """

    ATTRIBUTE_TYPES = (str, int, float, np.ndarray, "image")
    IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")

    def __init__(self, file_name: str, attributes: tuple, map_size: int,
                 show_image_name: bool = False,
                 encrypt: bool = False,
                 key_file_name: str = None):

        if not isinstance(file_name, str):
            raise TypeError("LMDB: file_name must be str")
        if not isinstance(attributes, (list, tuple)):
            raise TypeError("LMDB: attributes must be list/tuple")
        if not all(isinstance(attr, (str, bytes)) for attr in attributes):
            raise ValueError("LMDB: values in attributes must be str/bytes")
        attributes = [attr if isinstance(attr, str) else attr.encode()
                      for attr in attributes]
        if not isinstance(show_image_name, bool):
            raise TypeError("LMDB: show_image_name must be bool")
        if not isinstance(encrypt, bool):
            raise TypeError("LMDB: encrypt must be bool")
        if not (isinstance(key_file_name, str) or key_file_name is None):
            raise TypeError("LMDB: key_file_name must be str/None")
        if encrypt and not isinstance(key_file_name, str):
            raise ValueError("LMDB: key_file_name must be str")

        self.file_name = file_name
        self.attributes = attributes
        self.map_size = map_size * 4
        self.show_image_name = show_image_name
        self.encrypt = encrypt
        self.key_file_name = key_file_name
        self.n_samples = 0

    def __len__(self):
        return self.n_samples

    def __setitem__(self, key: bytes, value: bytes):
        assert isinstance(key, bytes) and isinstance(value, bytes)
        with self._env.begin(write=True) as f:
            f.put(key, value)
        return

    def __getitem__(self, key: bytes):
        assert isinstance(key, bytes)
        with self._env.begin(write=False) as f:
            content = f.get(key)
        return content

    def start(self, write: bool):
        r""" Starts the read/write lmdb environment! """
        if os.path.isfile(self.file_name):
            self.read_only = True
            self._env = lmdb.open(
                self.file_name, max_readers=1, readonly=not write, lock=False,
                readahead=False, meminit=False, subdir=False)
            self._load_len()
            self._load_attributes()
            self.encrypt = bool(self.__getitem__(b"encrypt").decode())
            if self.encrypt:
                self.__set_encrypt(False)
        elif write and not os.path.isfile(self.file_name):
            self._env = lmdb.open(
                self.file_name, map_size=self.map_size, subdir=False,
                readonly=False, meminit=False, map_async=True)
            self._note_len()
            self._note_attributes()
            self.__setitem__(b"encrypt", str(self.encrypt).encode())
            if self.encrypt:
                self.__set_encrypt(True)
        else:
            raise FileNotFoundError

    def stop(self):
        r""" Stops the read/write lmdb environment! """
        self._env.close()

    def read(self, idx: int):
        if not (0 <= idx < len(self)):
            print("LMDB: idx is not valid!")
            raise IndexError(repr(idx), "LMDB: idx is not valid, must be " +
                             "{}-{}!".format(0, len(self)-1))
        return self._decode(self.__getitem__("{:010}".format(idx).encode()))

    def write(self, *args):
        assert len(self.attributes) == len(args)
        self.__setitem__("{:010}".format(self.n_samples).encode(),
                         self._encode(args))
        self.n_samples += 1
        self._note_len()

    def _note_len(self):
        r""" Updates n_samples to the lmdb file """
        self.__setitem__(b"n_samples", str(int(self.n_samples)).encode())

    def _load_len(self):
        r""" Reads n_samples in the lmdb file """
        self.n_samples = int(self.__getitem__(b"n_samples").decode())

    def _note_attributes(self):
        r""" Writes attributes to the lmdb file """
        self.__setitem__(b"attributes", self._msgpack_encode(self.attributes))

    def _load_attributes(self):
        r""" Reads attributes in the lmdb file """
        self.attributes = self._msgpack_decode(self.__getitem__(b"attributes"))

    def _new_dict(self):
        r""" Standard format to store any sample in the database as bytes """
        return {b"type": None, b"content": None, b"dtype": b"",
                b"image_name": b""}

    def _attribute_encode(self, x):
        r""" Converts a value of type str/int/float/np.ndarray to bytes
            - str to bytes
            - int/float to str and then bytes
            - np.ndarray to base64 encoding
        """
        assert isinstance(x, LMDB.ATTRIBUTE_TYPES)
        out = self._new_dict()
        if isinstance(x, str):
            out[b"type"] = LMDB.ATTRIBUTE_TYPES.index(str)
            if x.lower().endswith(LMDB.IMAGE_TYPES):
                if not os.path.isfile(x):
                    raise ValueError("LMDB: Image does not exists!")
                x = x.encode()
                with open(x, "rb") as f:
                    content = f.read()
                if self.encrypt:
                    content = self.__encrypt.encrypt(content)
                    x = self.__encrypt.encrypt(x)
                out[b"content"] = content
                out[b"image_name"] = x
                out[b"type"] = LMDB.ATTRIBUTE_TYPES.index("image")
            else:
                out[b"content"] = x
        elif isinstance(x, int):
            out[b"type"] = LMDB.ATTRIBUTE_TYPES.index(int)
            out[b"content"] = str(x)
        elif isinstance(x, float):
            out[b"type"] = LMDB.ATTRIBUTE_TYPES.index(float)
            out[b"content"] = str(x)
        else:
            out[b"type"] = LMDB.ATTRIBUTE_TYPES.index(np.ndarray)
            out[b"content"] = base64.b64encode(x)
            out[b"dtype"] = x.dtype.str
        return out

    def _attribute_decode(self, x):
        r""" Converts bytes to a value of type str/int/float/np.ndarray """
        if x[b"type"] == LMDB.ATTRIBUTE_TYPES.index(str):
            return x[b"content"]
        elif x[b"type"] == LMDB.ATTRIBUTE_TYPES.index(int):
            return int(x[b"content"])
        elif x[b"type"] == LMDB.ATTRIBUTE_TYPES.index(float):
            return float(x[b"content"])
        elif x[b"type"] == LMDB.ATTRIBUTE_TYPES.index(np.ndarray):
            dtype = np.dtype(x[b"dtype"])
            return np.frombuffer(base64.decodebytes(x[b"content"]),
                                 dtype=dtype)
        else:
            if self.encrypt:
                image = ImPIL.open(io.BytesIO(
                    self.__encrypt.decrypt(x[b"content"])))
                name = self.__encrypt.decrypt(x[b"image_name"])
            else:
                image = ImPIL.open(io.BytesIO(x[b"content"]))
                name = x[b"image_name"]
            return (image, name.decode()) if self.show_image_name else image

    def _msgpack_encode(self, x):
        r""" Encodes content in x using msgpack to bytes """
        return msgpack.packb(x, use_bin_type=True)

    def _msgpack_decode(self, x):
        r""" Decodes x in bytes using msgpack """
        return msgpack.unpackb(x, raw=False, use_list=True)

    def _encode(self, values: (list, tuple)):
        r""" Encodes dictonary {attribute: value} to bytes, where each value
        of the attribute is encoded as per "_attribute_encode".
        Requirements:
            - len(attributes) == len(values), use None if no value
            - values must be list or tuple
            - a value must be one of LMDB.ATTRIBUTE_TYPES
        """
        content = {}
        for attribute, value in zip(self.attributes, values):
            if value is None:
                value = ""
            content[attribute] = self._attribute_encode(value)
        return self._msgpack_encode(content)

    def _decode(self, content: bytes):
        r""" Decodes bytes to a tuple of values for given attributes (order is
        same as attributes) - given the specific format better to read a lmdb
        file written by same function! """
        assert isinstance(content, bytes)
        content = self._msgpack_decode(content)
        values = []
        for attribute in self.attributes:
            value = self._attribute_decode(content[attribute])
            if value == "":
                value = None
            values.append(value)
        return tuple(values)

    def __set_encrypt(self, new_key: bool):
        from cryptography.fernet import Fernet
        if new_key:
            key = Fernet.generate_key()
            with open(self.key_file_name, "wb") as txt:
                txt.write(key)
        else:
            if not os.path.isfile(self.key_file_name):
                raise FileNotFoundError
            with open(self.key_file_name, "rb") as txt:
                key = txt.read()
        self.__encrypt = Fernet(key)
        del key


# os.remove("./test.lmdb")
# os.remove("./test.key")
# database = LMDB("./test.lmdb", ["np_arr", "label"], 1024*1024,
#                 encrypt=True, key_file_name="./test.key")
#
# database.start(True)
# database.write(np.random.randn(100), 0)
# database.write(np.random.randn(100), 1)
# database.write(np.random.randn(100), 2)
# database.write("../test.jpeg", 3)
# database.stop()
#
# database.start(False)
# database.read(0)[-1], database.read(1)[-1]
# database.read(2)[-1], database.read(3)[-1]
# database.read(4)[-1]
# database.show_image_name = False
# database.read(3)[0]
# database.show_image_name = True
# database.read(3)[0]
# len(database)
# database.stop()
