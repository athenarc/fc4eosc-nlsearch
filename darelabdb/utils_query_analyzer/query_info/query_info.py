from typing import Union, Literal, Any

from darelabdb.utils_query_analyzer.query_info.clause import (
    SetOperator,
    SelectClause,
    FromClause,
    WhereClause,
    GroupByClause,
    HavingClause,
    OrderByClause,
    LimitClause,
)
from darelabdb.utils_query_analyzer.query_info.operators import (
    Join,
    Aggregate,
    LogicalOperator,
    LikeOperator,
    ArithmeticOperator,
    ComparisonOperator,
    MembershipOperator,
    NullOperator,
    UnaryOperator,
    BinaryOperator,
)
from darelabdb.utils_query_analyzer.query_info.variables import Column, Table

# TODO add them in the class
CLAUSES = [
    SelectClause,
    FromClause,
    WhereClause,
    GroupByClause,
    HavingClause,
    OrderByClause,
    LimitClause,
]

OPERATOR_TYPES = [
    Join,
    Aggregate,
    ComparisonOperator,
    LogicalOperator,
    LikeOperator,
    ArithmeticOperator,
    MembershipOperator,
    NullOperator,
]

VARIABLE_TYPES = ["columns", "tables", "values"]


class QueryInfo:
    _clause_names = [
        "selectClause",
        "fromClause",
        "whereClause",
        "groupByClause",
        "havingClause",
        "orderByClause",
        "limitClause",
    ]

    def __init__(
        self,
        content: Union[
            SetOperator,
            tuple[
                SelectClause,
                FromClause,
                Union[WhereClause, None],
                Union[GroupByClause, None],
                Union[HavingClause, None],
                Union[OrderByClause, None],
                Union[LimitClause, None],
            ],
        ],
        alias: str = None,
        # database_info: DatabaseInfo = None,
    ):
        """
        Initializes the QueryInfo structure.
        Args:
            content: The clauses of the query
            # database_info (DatabaseInfo): The information about the database upon which the query is made. It is not
            #     required to be provided, but without it some information will be incomplete.
            #     Example: In the query 'select name from author join book on author.id = book.author_id where
            #     author.age < 18' the table of the column 'name' will be None due to lack of information.
        """

        self.alias = alias

        # Initialize the clauses
        if type(content) == SetOperator:
            self.selectClause = None
            self.fromClause = None
            self.whereClause = None
            self.groupByClause = None
            self.havingClause = None
            self.orderByClause = None
            self.limitClause = None
            self.setOperator = content
        else:
            self.selectClause = content[0]
            self.fromClause = content[1]
            self.whereClause = content[2]
            self.groupByClause = content[3]
            self.havingClause = content[4]
            self.orderByClause = content[5]
            self.limitClause = content[6]
            self.setOperator = None

            # Add implicit tables in columns
            # self._add_implicit_tables(database_info)
            self._add_implicit_tables()

    def __eq__(self, other: "QueryInfo") -> bool:
        if self.setOperator is not None:
            return other.setOperator == self.setOperator
        else:
            for clause in self._clause_names:
                if getattr(other, clause) != getattr(self, clause):
                    return False
            return True

    @classmethod
    def is_query(cls) -> bool:
        """Returns if the class can represent a query."""
        return True

    def _add_implicit_tables(self) -> None:
        """
        Adds the table name in the columns that are not explicitly mentioned.
        Without the schema implicit tables can be added only in case of 1 table in the 'from' clause!.
        """
        # If the information about the database are not provided and the query has only 1 table
        # if database_info is None:
        # TODO better
        query_tables = self.fromClause._tables
        # If there is only one table in the query
        if len(query_tables) == 1 and not isinstance(query_tables[0], Join):
            implicit_table_name = (
                query_tables[0].name
                if isinstance(query_tables[0], Table)
                else query_tables[0].alias
            )
            # Add the table name in the columns that is not explicitly stated
            for column in self.columns(return_format="raw", shallow_search=True):
                # If the table name
                if column.table_name is None:
                    column.table_name = implicit_table_name

    def get_columns_equivalences(
        self, shallow_search: bool = True
    ) -> dict[str, list[str]]:
        """
        Returns the columns in the query that can be used interchangeable due to equality condition. The columns are
        stored in the dictionary with their fullname (`<table_name>.<column_name>`), where table_name is the actual name
        and not an alias, if the there is information about the column's table else it is omitted.

        !!! warning "shallow_search"

            Currently, the shallow_search = False is not supported!

        Args:
            shallow_search (bool): If True the columns of the subqueries will not be
                considered.

        Returns:
            columns: A dictionary with the names of the columns and a list with their equivalences.
        """

        aliases = self.aliases(shallow_search=shallow_search)
        tables = self.tables(shallow_search=shallow_search, unique=True)

        # Get the equality conditions of the query
        comp_ops = self.comparison_operators(
            shallow_search=shallow_search, return_format="raw"
        )
        eq_ops = [op for op in comp_ops if op.op]

        # Create an equivalences dictionary based on the equality conditions
        equivalences = {}
        for eq_op in eq_ops:
            operands = eq_op.operands
            # If the operands are both columns
            if isinstance(operands[0], Column) and isinstance(operands[1], Column):
                columns = self._replace_table_aliases(operands, aliases, tables)

                if columns[0] in equivalences:
                    equivalences[columns[0]].append(columns[1])
                else:
                    equivalences[columns[0]] = [columns[1]]

                if columns[1] in equivalences:
                    equivalences[columns[1]].append(columns[0])
                else:
                    equivalences[columns[1]] = [columns[0]]

        return equivalences

    def move_join_on_condition_in_where(self) -> None:
        if self.setOperator:
            for query in self.setOperator.queries:
                query.move_join_on_condition_in_where()
        else:
            # Move the 'on' conditions of joins in the current depth
            for table in self.fromClause._tables:
                if type(table) is Join and table._on_condition is not None:
                    condition = table._on_condition
                    table._on_condition = None
                    if self.whereClause is not None:
                        self.whereClause._condition = LogicalOperator(
                            op="and", operands=(self.whereClause._condition, condition)
                        )
                    else:
                        self.whereClause = WhereClause(condition=condition)
            # Move the 'on' conditions of join in the subqueries
            for query in self.subqueries():
                query.move_join_on_condition_in_where()

    @staticmethod
    def _replace_table_aliases(
        columns: list[Column], aliases: dict, tables: list[str]
    ) -> list[str]:
        columns_without_aliases = []
        for column in columns:
            # If the table in the column is not in the schema tables and exists in the aliases of the query
            if (
                column.table_name is not None
                and column.table_name not in tables
                and column.table_name in aliases
            ):
                columns_without_aliases.append(
                    f"{aliases[column.table_name]}.{column.name}"
                )
            else:
                columns_without_aliases.append(column.fullname)
        return columns_without_aliases

    def columns_without_table_aliases(
        self, shallow_search: bool = False, unique: bool = False
    ) -> list[str]:
        """
        Replace the table name in the columns if it is an alias to a schema table and returns the
        `<table_name>.<column_name>` for each column in the query, if the information about the table are not available
        only the column name is returned.

        !!! warning "Query as Table"

            In case of a query as table the alias remains.

        !!! warning "Aliases in subqueries"

            Current version does not support alias visibility to other queries (inner or outer).

        Args:
            shallow_search (bool): If True the columns of the subqueries are not included in the returned list.
            unique (bool): If True only the unique values are kept.

        Returns:
            columns: A list with the columns names and its table.
        """
        aliases = self.aliases(shallow_search=True)
        tables = self.tables(shallow_search=True, unique=True)

        cols = self._replace_table_aliases(
            self.columns(return_format="raw", shallow_search=True, unique=unique),
            aliases,
            tables,
        )

        if not shallow_search:
            for subquery in self.subqueries():
                cols.extend(
                    subquery.columns_without_table_aliases(
                        shallow_search=False, unique=unique
                    )
                )

        if unique:
            cols = list(set(cols))

        return cols

    def _get_property_list(
        self, property_name: str, clauses: list[str] = None, **kwargs
    ) -> list[Any]:
        """
        Creates a list with the values of the property in the requested clauses.
        Args:
            property_name (str): The name of the property that will be accessed.
            clauses (list(str)): A list of clauses that will access for the property values. If None,
                all clauses are considered.
            shallow_search (bool): If True the structural components of the subqueries will not be
                considered.
        """
        clause_names = (
            self._clause_names + ["setOperator"] if clauses is None else clauses
        )

        property_values = []
        for clause_name in clause_names:
            clause = getattr(self, clause_name)
            if clause is not None and hasattr(clause, property_name):
                property_values.extend(getattr(clause, property_name)(**kwargs))
        return property_values

    def _get_property_dict(self, property_name: str, **kwargs) -> dict:
        property_values = {}
        for clause_name in self._clause_names + ["setOperator"]:
            clause = getattr(self, clause_name)
            if clause is not None and hasattr(clause, property_name):
                property_values.update(getattr(clause, property_name)(**kwargs))
        return property_values

    def subqueries(self) -> list["QueryInfo"]:
        """Returns a list with the subqueries existing in the query."""
        return self._get_property_list("subqueries")

    def aliases(self, shallow_search: bool = False) -> dict[str, str]:
        """
        Creates a dictionary with the aliases and the corresponding schema element names. If the alias is not for
        a schema element the value is None.
        Args:
            shallow_search:

        Returns:
            aliases: A dictionary with the aliases and the actual schema element names.
        """
        aliases = self._get_property_dict(
            "aliases", **{"shallow_search": shallow_search}
        )
        if self.alias is not None:
            aliases[self.alias] = None
        return aliases

    def joins(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, Join]:
        """
        Retrieves the joins existing in the query.

        Args:
            shallow_search (bool): If True the joins of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            joins: A list with the join operators existing in the query.
        """
        return self._get_property_list(
            "joins",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def aggregates(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, Aggregate]:
        """
        Retrieves the aggregates existing in the query.

        Args:
            shallow_search (bool): If True the aggregates of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            aggregates: A list with the aggregate operators existing in the query.
        """
        return self._get_property_list(
            "aggregates",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def comparison_operators(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, ComparisonOperator]:
        """
        Retrieves the comparison operators existing in the query.

        Args:
            shallow_search (bool): If True the comparison operators of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            comparison_operators: A list with the comparison operators existing in the query.
        """
        return self._get_property_list(
            "comparison_operators",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def logical_operators(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, LogicalOperator]:
        """
        Retrieves the logical operators existing in the query.

        Args:
            shallow_search (bool): If True the logical operators of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            logical_operators: A list with the logical operators existing in the query.
        """
        return self._get_property_list(
            "logical_operators",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def like_operators(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, LikeOperator]:
        """
        Retrieves the like operators existing in the query.

        Args:
            shallow_search (bool): If True the like operators of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            like_operators: A list with the like operators existing in the query.
        """
        return self._get_property_list(
            "like_operators",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def arithmetic_operators(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, ArithmeticOperator]:
        """
        Retrieves the arithmetic operators existing in the query.

        Args:
            shallow_search (bool): If True the arithmetic operators of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            arithmetic_operators: A list with the arithmetic operators existing in the query.
        """
        return self._get_property_list(
            property_name="arithmetic_operators",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def membership_operators(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, MembershipOperator]:
        """
        Retrieves the membership operators existing in the query.

        Args:
            shallow_search (bool): If True the membership of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            membership_operators: A list with the membership operators existing in the query.
        """
        return self._get_property_list(
            "membership_operators",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def null_operators(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, NullOperator]:
        """
        Retrieves the null operators existing in the query.

        Args:
            shallow_search (bool): If True the null operators of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            null_operators: A list with the null operators existing in the query.
        """
        return self._get_property_list(
            "null_operators",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def unary_operators(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, UnaryOperator]:
        """
        Retrieves the unary operators existing in the query.

        Args:
            shallow_search (bool): If True the unary operators of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            unary_operators: A list with the unary operators existing in the query.
        """
        return self._get_property_list(
            "unary_operators",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def binary_operators(
        self,
        shallow_search: bool = False,
        return_format: Literal["name", "raw"] = "name",
    ) -> list[str, BinaryOperator]:
        """
        Retrieves the binary operators existing in the query.

        Args:
            shallow_search (bool): If True the binary operators of the subqueries will not be
                considered.
            return_format (Literal["name", "raw"] = "name"): The format of the returned operators.

                - `name`: The name of each operator is returned.
                - `raw`: The class of each operator is returned.

        Returns:
            binary_operators: A list with the binary operators existing in the query.
        """
        return self._get_property_list(
            "binary_operators",
            **{"shallow_search": shallow_search, "return_format": return_format},
        )

    def tables(self, shallow_search: bool = False, unique: bool = False) -> list[str]:
        """
        Retrieves the names of the schema tables existing in the query.

        Args:
            shallow_search (bool): If True the tables of the subqueries will not be
                considered.
            unique (bool): If True only the unique values are kept.

        Returns:
            tables: A list with the tables in the query.
        """

        t = self._get_property_list("tables", **{"shallow_search": shallow_search})

        if unique:
            t = list(set(t))

        return t

    def columns(
        self,
        return_format: Literal["name", "complete_name", "raw"] = "complete_name",
        shallow_search: bool = False,
        unique: bool = False,
    ) -> list[Union[str, Column]]:
        """
        Retrieves the columns existing in the query.

        Args:
            return_format (Literal["name", "complete_name", "raw"]): The format of the returned columns.

                - `name`: Only the names of the columns are returned
                - `complete_name`: The name of the table followed by the column name is returned for each column
                - `raw`: A class of type Column is returned for each column

            shallow_search (bool): If True the columns of the subqueries will not be
                considered.
            unique (bool): If True only the unique values are kept.

        Returns:
            columns: The columns existing in the query.
        """

        cols = self._get_property_list(
            "columns",
            **{
                "shallow_search": shallow_search,
                "return_format": return_format,
                "unique": unique,
            },
        )

        # Filter alias that are found as columns
        for alias in list(self.aliases().keys()):
            match return_format:
                case "name":
                    cols = list(filter(lambda x: not x.endswith(alias), cols))
                case "complete_name":
                    cols = list(
                        filter(
                            lambda x: not (
                                x.endswith(alias)
                                or ("." in x and x.endswith(f".{alias}"))
                            ),
                            cols,
                        )
                    )
                case "raw":
                    cols = list(filter(lambda x: not x.name.endswith(alias), cols))

        if unique:
            cols = list(set(cols))

        return cols

    def values(self, shallow_search: bool = False) -> list[Any]:
        """
        Retrieves the values (e.g., literals, numbers) existing in the query.

        Args:
            shallow_search (bool): If True the values of the subqueries will not be
                considered.

        Returns:
            values: The values existing in the query.
        """
        return list(
            self._get_property_list("values", **{"shallow_search": shallow_search})
        )

    def setOperators(self, shallow_search: bool = False) -> list["SetOperator"]:
        return self._get_property_list(
            "setOperators", **{"shallow_search": shallow_search}
        )

    def depth(self) -> int:
        """
        Calculates the depth of the query.

        Returns:
            depth: The depth of the query
        """
        return max([subquery.depth() + 1 for subquery in self.subqueries()] + [0])

    def structural_components(
        self, shallow_search: bool = False, with_names: bool = False, with_pos=False
    ) -> list[str]:
        """
        Creates a list with the structural components existing in the query.

        !!! info "Structural components"

            select(S), from(F), where(W), group by(G), having(H), order by(O), limit(L),
            set operator(SO), nesting(N)

        Args:
            shallow_search (bool): If True the structural components of the subqueries will not be considered.
            with_names (bool): If true the structural components are returned with their names, else with their
                abbreviations.
            with_pos (bool): If True the nesting is returned as <N_pos> or <nesting_pos>, where pos is the clause in
                which the nesting appears.

        Returns:
            structural_components: A list with the structural components existing in the query.
        """
        components = []

        # If the query is a set operator
        if self.setOperator is not None:
            # Add the set operator in the structural components
            components.append(
                self.setOperator.class_name() if with_names else self.setOperator.abbr()
            )
            # Add the structural components of the queries of the set operator
            for query in self.setOperator.queries:
                components.extend(
                    query.structural_components(
                        shallow_search=True, with_names=with_names, with_pos=with_pos
                    )
                )
        else:
            # Add the clauses in the structural components
            for clause_name in self._clause_names:
                clause = getattr(self, clause_name)
                if clause is not None:
                    components.append(
                        clause.class_name() if with_names else clause.abbr()
                    )

            # Add the nestings in the structural components
            if with_pos:
                for clause_name in self._clause_names:
                    clause = getattr(self, clause_name)
                    if clause is not None and hasattr(clause, "subqueries"):
                        subqueries = clause.subqueries()
                        if len(subqueries):
                            components.extend(
                                (
                                    f"nesting_{clause_name}"
                                    if with_names
                                    else f"N_{clause_name}"
                                )
                                for _ in range(len(subqueries))
                            )
            else:
                nestings_num = len(self.subqueries()) - len(self.setOperators())
                components.extend(
                    ["nesting" if with_names else "N" for _ in range(nestings_num)]
                )

        if not shallow_search:
            # Add the structural components of each subquery
            for subquery in self.subqueries():
                components.extend(
                    subquery.structural_components(
                        shallow_search=False, with_names=with_names, with_pos=with_pos
                    )
                )

        return components

    def operator_types(
        self,
        shallow_search: bool = False,
        with_names: bool = False,
        per_clause: bool = False,
    ) -> Union[list[str], dict[str, list[str]]]:
        """
        Returns all the operator types existing in the query.
        If the `per_clause` parameter is True the operators are returned per clause.

        !!! info "Operator types"

            joins(J), aggregates(Ag), comparison_operators(C), logical_operators(Lo), like_operators(Li),
            arithmetic_operators(Ar), membership_operators(Me), null_operators(Nu), unary_operators(UnOp),
            binary_operators(BiOp)

        Args:
            shallow_search (bool): If True the structural components of the subqueries will not be
                considered.
            with_names (bool): If true the structural components are returned with their names, else with their
                abbreviations.
            per_clause (bool): If True the method returns the operator types per clause.

        Returns:
            operator_types: A list with the operator types existing in the query or a dictionary with the operator types
                per clause.

        """
        operatorTypes = {} if per_clause else []

        for clause_name in self._clause_names + ["setOperator"]:
            clause = getattr(self, clause_name)
            if clause is not None:
                if per_clause:
                    operatorTypes[clause.class_name()] = clause.operator_types(
                        shallow_search=shallow_search, with_names=with_names
                    )
                else:
                    operatorTypes.extend(
                        clause.operator_types(
                            shallow_search=shallow_search, with_names=with_names
                        )
                    )

        return operatorTypes

    def operators(
        self, shallow_search: bool = False, per_clause: bool = False
    ) -> Union[list[str], dict[str, list[str]]]:
        """
        Retrieves all the operators existing in the query.

        Args:
            shallow_search: If True the operators of the subqueries will not be
                considered.
            per_clause: If True the method returns the operators per clause.

        Returns:
            operators: A list with the operators of the query or a dictionary with the operators of the query per
                clause.
        """
        operators = {} if per_clause else []

        for clause_name in self._clause_names + ["setOperator"]:
            clause = getattr(self, clause_name)
            if clause is not None:
                if per_clause:
                    operators[clause.class_name()] = clause.operators(
                        shallow_search=shallow_search
                    )
                else:
                    operators.extend(clause.operators(shallow_search=shallow_search))
        return operators

    @staticmethod
    def _get_category(
        category_components: list[str],
        query_components: list[str],
        return_format: Literal["binary", "counter", "name"],
    ):
        """
        Returns the category of the query based on the category components and the components found in the query.

        Args:
            category_components (list[str]): The components that comprise the category.
            query_components (list[str]): The components found in the query.
            return_format (str): The format in which the category will be returned.

                - 'binary': Each category's component will be 0 or 1, depending on the existence or not in the query.
                - 'counter': Each category's component will represented by the number of appearances in the query.
                - 'name': The category will contain the name of a component if the component appears in the query.
        """
        category = []

        for component in category_components:
            if return_format == "counter":
                category.append(str(query_components.count(component)))
            else:
                component_exists = component in query_components

                if return_format == "binary":
                    category.append(str(1 if component_exists else 0))
                else:  # If return_format == "name"
                    if component_exists:
                        category.append(component)

        if return_format == "binary":
            return "".join(category)
        elif return_format == "name":
            return "-".join(category)
        else:  # if return_format == "counter"
            # sep is added for counter values > 9
            return ".".join(category)

    def get_structural_category(
        self,
        return_format: Literal["binary", "counter", "name"] = "name",
        shallow_search: bool = False,
    ) -> str:
        """
        Creates the structural category of the query. The structural category is defined as the set of the structural
        components existing in the query.

        !!! info "Structural components"

            select(S), from(F), where(W), group by(G), order by(O), having(H), limit(L),
            set operator(SO), nesting(N)

        Args:
            return_format (str): The format in which the category will be returned.

                - `binary`: Each category's component will be 0 or 1, depending on the existence or not of a structural
                    component.
                - `counter`: Each category's component with be the number of appearances of the corresponding structural
                    component.
                - `name`: The category will contain the abbreviations of a structural component if the component
                    appears in the query.
            shallow_search (bool): If True the structural components of the subqueries will not be
                considered.

        Returns:
            structural_category: The structural category of the query.

        """
        query_structural_components = self.structural_components(
            shallow_search=shallow_search
        )

        structural_components_abbrs = [
            SelectClause.abbr(),
            FromClause.abbr(),
            WhereClause.abbr(),
            GroupByClause.abbr(),
            HavingClause.abbr(),
            OrderByClause.abbr(),
            LimitClause.abbr(),
            SetOperator.abbr(),
            "N",
        ]

        return self._get_category(
            category_components=structural_components_abbrs,
            query_components=query_structural_components,
            return_format=return_format,
        )

    def get_operator_types_category(
        self,
        return_format: Literal["binary", "counter", "name"],
        shallow_search: bool = False,
    ) -> str:
        """
        Creates the operator types category of the query. The operator types category is defined as the set of the
        operator types existing in the query.

        !!! info "Operator types"

            joins(J), aggregates(Ag), comparison_operators(C), logical_operators(Lo), like_operators(Li),
            arithmetic_operators(Ar), membership_operators(Me), null_operators(Nu), unary_operators(UnOp),
            binary_operators(BiOp)

        Args:
            return_format: The format in which the category will be returned.

                - `binary`: Each category's component will be 0 or 1, depending on the existence or not of an operator type.
                - `counter`: Each category's component with be the number of appearances of the corresponding operator type.
                - `name`: The category will contain the name of an operator type if it appears in the query.
            shallow_search (bool): If True the structural components of the subqueries will not be
                                                   considered.

        Returns:
            operator_types_category: The operator types category of the query.
        """
        query_operatorTypes = self.operator_types(shallow_search=shallow_search)

        # TODO add abbreviations in the operators' classes
        operatorTypes_abbrs = [
            "J",
            "Ag",
            "C",
            "Lo",
            "Li",
            "Ar",
            "Me",
            "Nu",
            "UnOp",
            "BiOp",
        ]

        return self._get_category(
            category_components=operatorTypes_abbrs,
            query_components=query_operatorTypes,
            return_format=return_format,
        )


def get_subqueries(queries: list[QueryInfo]) -> list[QueryInfo]:
    """
    Returns a list with the subqueries of all the given queries.
    """
    subs = []
    for query in queries:
        subs.extend(query.subqueries())
    return subs


def get_subqueries_per_depth(query_info: QueryInfo) -> dict[int, list]:
    """
    Returns a dictionary with the depth and the subqueries that correspond the depth in the given query.

    Args:
        query_info (QueryInfo): The given query_info class
    """
    depth = 0
    subqueries_per_depth = [[query_info]]

    while len(subqueries_per_depth[-1]) > 0:
        subqueries_per_depth.append(get_subqueries(subqueries_per_depth[-1]))

    # Remove the last entry which has 0 subqueries
    subqueries_per_depth = subqueries_per_depth[:-1]

    return {
        depth: depth_subqueries
        for depth, depth_subqueries in enumerate(subqueries_per_depth)
    }
