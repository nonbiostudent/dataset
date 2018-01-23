import hashlib
import warnings

import numpy as np
import tables
from tables.exceptions import NoSuchNodeError, NodeError

from dataset.class_factory import ResourceIdentifier
from dataset.plugins import get_registered_plugins

_all_classes = None




class Dataset(object):
    """
    This class is a container for all data describing a spectroscopy analysis
    from the raw measurements, over instruments and information on gas plumes
    to the final gas flux results.

    :type preferredFluxIDs: list
    :param preferredFluxIDs: IDs of the best/final flux estimate. As a dataset
        can contain analyses from different targets, there can be more than one
        preferred flux estimate.
    :type spectra: list
    :param spectra: List of all spectra that are part of the dataset.
    :type instruments: list
    :param instruments: List of all instruments that are part of the dataset.
    :type retrievals: list
    :param retrievals: List of all retrievals that are part of the dataset.
    :type plumevelocities: list
    :param plumevelocities: List of all plume velocities that are part of the
        dataset.
    :type targets: list
    :param targets: List of all target plumes that are part of the dataset.
    :type flux: list
    :param flux: List of all flux estimates that are part of the dataset.
    """

    def __init__(self, filename, mode):
        
        if _all_classes is None:
            raise ValueError("dataset.set_datamodel() must be called prior to " 
                             "creating a Dataset object")
        
        self.elements = {}
        self.base_elements = {}
        for c in _all_classes:
            name = c.__name__.strip('_') 
            self.elements[c.__dest__] = []
            #if self.elements.__dict__.has_key(c.__dest__):
             #   raise ValueError("Forbidden element destination %s"%c.__dest__)
            
            #self.elements.__dict__[c.__dest__] = self.elements[c.__dest__]
            self.base_elements[name] = c
        self._rids = {}
        self._f = tables.open_file(filename, mode)
        # Create an array of sha224 hash values; when
        # opening an existing file this will throw an
        # exception
        try:
            self._f.create_earray('/','hash',tables.StringAtom(itemsize=28),(0,))
        except NodeError:
            pass

    def __del__(self):
        self._f.close()
    
    def __add__(self, other):
        msg = "__add__ is undefined as the return value would "
        msg += "be a new hdf5 file with unknown filename."
        raise AttributeError(msg)                             

    def __iadd__(self, other):
        if self._f == other._f:
            raise ValueError("You can't add a dataset to itself.")
        update_refs = []
        rid_dict = {}
        for e in other.elements.keys():
            for k in other.elements[e]:
                ne = self._copy_children(k)
                self.elements[e].append(ne)
                update_refs.append(ne)
                rid_dict[str(k._resource_id)] = str(ne._resource_id)

        for ne in update_refs:
            for k in ne._reference_keys:
                table = getattr(ne._root,'data',None)
                if table is not None:
                    ref = getattr(ne._root.data.cols,k,None)
                    if ref is not None:
                        if type(ref[0]) == np.ndarray:
                            newentry = []
                            for iref in ref[0]:
                                newentry.append(rid_dict[iref])
                            ref[0] = np.array(newentry)
                        else:
                            ref[0] = rid_dict[ref[0]]
        return self

    def _newdst_group(self, dstgroup, title='', filters=None):
        """
        Create the destination group in a new HDF5 file.
        """
        group = self._f.root
        # Now, create the new group. This works even if dstgroup == '/'
        for nodename in dstgroup.split('/'):
            if nodename == '':
                continue
            # First try if possible intermediate groups already exist.
            try:
                group2 = self._f.get_node(group, nodename)
            except NoSuchNodeError:
                # The group does not exist. Create it.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    group2 = self._f.create_group(group, nodename,
                                                title=title,
                                                filters=filters)
            group = group2
        return group

    def _copy_children(self, src, title='', recursive=True,
                       filters=None, copyuserattrs=False,
                       overwrtnodes=False):
        """
        Copy the children from source group to destination group
        """
        srcgroup = src._root
        # assign a new resource ID so that both objects can 
        # be referred to within the same session
        dstgroup = srcgroup._v_parent._v_pathname+'/'+ str(ResourceIdentifier())
        created_dstgroup = False
        # Create the new group
        dstgroup = self._newdst_group(dstgroup, title, filters)

        # Copy the attributes to dstgroup, if needed
        if copyuserattrs:
            srcgroup._v_attrs._f_copy(dstgroup)

        # Finally, copy srcgroup children to dstgroup
        try:
            srcgroup._f_copy_children(
                dstgroup, recursive=recursive, filters=filters,
                copyuserattrs=copyuserattrs, overwrite=overwrtnodes)
        except:
            msg = "Problems doing the copy of '{:s}'.".format(dstgroup)
            msg += "Please check that the node names are not "
            msg += "duplicated in destination, and if so, enable "
            msg += "overwriting nodes if desired."
            raise RuntimeError(msg)
        return type(src)(dstgroup)

    def new(self, data_buffer, pedantic=True):
        """
        Create a new entry in the HDF5 file from the given data buffer.
        """
        if pedantic:
            s = hashlib.sha224()
            # If data buffer is empty raise an exception
            incomplete = False
            for k,v in data_buffer.__dict__.iteritems():
                if k == 'tags':
                    continue
                if v is not None:
                    if k in data_buffer._property_dict.keys():
                        s.update('{}'.format(v))
                else:
                    incomplete = True
            if incomplete:
                raise ValueError("You cannot add incomplete buffers if 'pedantic=True'.")

        _C = self.base_elements[type(data_buffer).__name__[:-6]] #strip 'Buffer' suffix
        group_name = _C.__dest__#_C.__name__.strip('_')
        rid = ResourceIdentifier()
        try:
            self._f.create_group('/',group_name)
        except tables.NodeError:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            group = self._f.create_group('/'+group_name,str(rid))
        e = _C(group,data_buffer, pedantic=pedantic)
        self.elements[group_name].append(e)
        return e         

    def read(self, filename, ftype, **kwargs):
        """
        Read in a datafile.
        """
        plugins = get_registered_plugins()
        pg = plugins[ftype.lower()]()
        return pg.read(self,filename,**kwargs)
    
    @staticmethod
    def open(filename):
        """
        Open an existing HDF5 file.
        """
        dnew = Dataset(filename,'r+')
        
        valid_names = [n[:-6] for n in dnew.base_elements] #names without "Buffer" suffix
        
        dest_name_map = {} #mapping between __dest__ values and class names
        
        #add in any destinations defined in the UML
        for c in dnew.base_elements.values():
            try:
                valid_names.append(c.__dest__)
                name = c.__name__.strip('_')
                dest_name_map[c.__dest__] = name
            except KeyError:
                continue
                 
        for group in dnew._f.walk_groups('/'):
            if group._v_name is '/' or group._v_name not in valid_names:
                continue
            for sgroup in group._v_groups.keys():
                class_name = dest_name_map[group._v_name]
                
                _C = dnew.base_elements[class_name]
                e = _C(group._v_groups[sgroup])
                print "appending to %s"%group._v_name
                dnew.elements[group._v_name].append(e)
        return dnew

    def close(self):
        """
        Close the HDF5 file and clear the ResourceIdentifiers.
        """
        for g in self.elements:
            for e in self.elements[g]:
                del e._resource_id
        self._f.close()

    def register_tags(self, tags):
        """
        Register one or more tag names.
        """
        try:
            self._f.create_group('/','tags')
        except NodeError:
            pass
        for tag in tags:
            try:
                self._f.create_earray('/tags', tag, tables.StringAtom(itemsize=60), (0,))
            except NodeError:
                raise ValueError("Tag '{:s}' has already been registered".format(tag))

    def remove_tags(self, tags):
        """
        Remove one or more tag names. This will also remove the tag from every
        element that had been tagged.
        """
        for tag in tags:
            try:
                ea = self._f.root.tags._v_children[tag]
                for rid in ea[:]:
                    e = ResourceIdentifier(rid).get_referred_object()
                    e.tags.remove(tag)
            except (KeyError, NoSuchNodeError):
                warnings.warn("Can't remove tag {} as it doesn't exist.".format(tag)) 

    def select(self, *args, **kargs):
        """
        Find a subset of events based on given select rules.
        """
        try:
            etype = kargs['etype']
        except KeyError:
            etype = False

        if etype:
            retval = []
            for _e in self.elements[etype]:
                pass

                
           
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)

