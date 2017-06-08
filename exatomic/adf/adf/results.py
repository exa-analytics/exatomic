
parser_aliases = {'EFG': EFG}

class RESULTS(Sections):
    """
    Sections parser for the 'R E S U L T S' sub-section of an 'A D F' calculation.
    """
    _key_delim = "^[ ]+=+$"
    _key_start_minus = -2
    _key_starts = 0
    _key_names = "_INFO_"
    _key_re_name = re.compile("([A-z]+)")

    def _parse(self):
        """
        """
        delims = self.regex(self._key_delim, text=False)[self._key_delim]
        starts = [self._key_starts]
        names = [self._key_names]
        titles = [self._key_names]
        ends = []
        for i in delims:
            start = i + self._key_start_minus
            search = self._key_re_name.search(str(self[start]))
            if search is not None and len(search.groups()) >= 1:
                name = str(self[start]).strip()
                for key, prsr in parser_aliases.items():
                    if key in name:
                        names.append(prsr)
                        break
                else:
                    names.append(name)
                starts.append(start)
                ends.append(start)
                titles.append(name)
        ends.append(len(self))
        self._sections_helper(parser=names, start=starts, end=ends, title=titles)
