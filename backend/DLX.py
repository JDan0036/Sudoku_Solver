class DLX:
    PRIMARY = 0
    SECONDARY = 1
    def __init__(self, columns, rows=None, rowNames=None):
        self.node_count = len(columns) + 1 # normal columns + header
        
        # Initialized vertical traversal within a column
        self.U = [i for i in range(self.node_count)]
        self.D = [i for i in range(self.node_count)]
        # Initialized horizontal traversal within a row
        self.L = [0] * self.node_count
        self.R = list(range(self.node_count))

        self.C = list(range(self.node_count))    # Which column this node belongs to
        self.S = [0] * self.node_count  # Size of the columns
        self.N = [colname for (colname,_) in columns] + [None]  # Names of the columns

        self._link_primary_column(columns, rows, rowNames)

    def _link_primary_column(self, columns, rows, rowNames):
        '''
        Links the L and R pointers so that only primary columns are linked together.
        '''
        previous = self.node_count - 1
        current = 0

        # Builds the L-links of primary columns together by skipping secondary columns.
        for (colname, coltype) in columns:
            if coltype == DLX.PRIMARY:
                self.L[current] = previous
                previous = current
            else:
                self.L[current] = current
            current += 1
        self.L[self.node_count - 1] = previous

        # Now convert those L-links into matching R-links
        previous = self.node_count-1
        cur = self.L[previous]
        while cur != self.node_count-1:
            self.R[cur] = previous
            previous = cur
            cur = self.L[cur]
        self.R[self.node_count-1] = previous

        # Store the header index.
        self.header = len(columns)

        # Store the solution variable.
        self.partialsolution = []

        # If there are any rows, append them.
        if rows:
            self.appendRows(rows, rowNames)

    def _solve(self, columnselector, userdata):
        # Solution complete
        if self.R[self.header] == self.header:
            return True

        # Choose column
        c = columnselector(self, userdata)
        if c == self.header or self.S[c] == 0:
            return False

        self._cover(c)

        r = self.D[c]
        while r != c:
            self.partialsolution.append(r)

            j = self.R[r]
            while j != r:
                self._cover(self.C[j])
                j = self.R[j]

            if self._solve(columnselector, userdata):
                return True

            # Backtrack
            self.partialsolution.pop()
            j = self.L[r]
            while j != r:
                self._uncover(self.C[j])
                j = self.L[j]

            r = self.D[r]

        self._uncover(c)
        return False
    
    def _cover(self, c):
        # Remove this column from the header.
        self.L[self.R[c]] = self.L[c]
        self.R[self.L[c]] = self.R[c]

        # Iterate over the rows covered by this column.
        # Stop when we reach the header.
        i = self.D[c]
        while i != c:
            # Remove this row from the problem.
            j = self.R[i]
            while j != i:
                self.U[self.D[j]] = self.U[j]
                self.D[self.U[j]] = self.D[j]
                self.S[self.C[j]] -= 1
                j = self.R[j]
            i = self.D[i]

    def _uncover(self, c):

        # Reverse the operations done in _cover.
        i = self.U[c]
        while i != c:
            j = self.L[i]
            while j != i:
                self.S[self.C[j]] += 1
                self.D[self.U[j]] = j
                self.U[self.D[j]] = j
                j = self.L[j]
            i = self.U[i]

        # Readd the column to the header.
        self.R[self.L[c]] = c
        self.L[self.R[c]] = c

    def appendRows(self, rows, rowNames=None):
        if rowNames == None:
            rowNames = [None] * len(rows)

        rowIds = []
        
        for i in range(len(rows)):
            rowIds.append(self.appendRow(rows[i], rowNames[i]))
        return rowIds
    
    def appendRow(self, row, rowName=None):

        first = None
        previous = None
        
        for index in row:
            node = self.node_count

            # ---- Column (vertical) insertion ----
            self.C.append(index)
            self.S[index] += 1

            self.U.append(self.U[index])
            self.D.append(index)
            self.D[self.U[index]] = node
            self.U[index] = node

            # ---- Row (horizontal) insertion ----
            if first is None:
                # first node in the row
                first = node
                self.L.append(node)
                self.R.append(node)
            else:
                # link after prev
                self.L.append(prev)
                self.R.append(first)
                self.R[prev] = node
                self.L[first] = node

            # ---- Metadata ----
            self.N.append(rowName)

            prev = node
            self.node_count += 1
        return first

    def smallestColumnSelector(self, _):
        """Select the column with the fewest rows covering it, i.e. minimize
        the branching factor.

        Note that the userdata (second parameter) is ignored."""

        smallest = self.R[self.header]
        j = self.R[self.R[self.header]]
        while j != self.header:
            if self.S[j] < self.S[smallest]:
                smallest = j
            j = self.R[j]
        return smallest

    def solve(self, columnselector=smallestColumnSelector, columnselectoruserdata=None):
        self.partialsolution = []
        if columnselector is None:
            columnselector = DLX.smallestColumnSelector

        found = self._solve(columnselector, columnselectoruserdata)

        if found:
            return self.partialsolution[:]  # return copy
        else:
            return None


