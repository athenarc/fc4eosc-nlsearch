from typing import Literal

COMPACT_SCHEMA = """\
result(description, id, language, type, title, publisher, publication_date, country, accessright, keywords),
author(fromorcid, sk_id, lastname, orcid, fullname, firstname),
result_author(sk_author_id, sk_result_id), -- used to join result and author
result_citations(id, sk_result_id_cited, sk_result_id_cites), -- used to join result and citations
community(sk_id, name, acronym, description),
result_community(sk_community_id, sk_result_id) -- used to join result and community
fos(sk_id, label),
result_fos(sk_result_id, sk_fos_id) -- used to join result and fos
"""

DDL_SCHEMA = """\
CREATE TABLE result (
    sk_id int primary key,
    accessright varchar(100),
    country varchar(50),
    description text,
    keywords varchar(200),
    language varchar(100),
    publication_date date,
    publisher varchar(100),
    title varchar(200),
    type varchar(50)
);

CREATE TABLE author (
    sk_id int primary key,
    fromOrcid varchar(10),
    fullname varchar(500),
    orcid varchar(20)
);

CREATE TABLE result_author (
    sk_author_id int,
    sk_result_id int,
    primary key (sk_author_id, sk_result_id)
);

CREATE TABLE result_citations (
    id varchar(200) primary key,
    sk_result_id_cited int,
    sk_result_id_cites int
);

CREATE TABLE community (
    sk_id int primary key,
    name varchar(100),
    acronym varchar(20),
    description varchar(1000)
);

CREATE TABLE result_community (
    sk_community_id int,
    sk_result_id int,
    primary key (sk_community_id, sk_result_id)
);

CREATE TABLE result_community (
    sk_community_id int,
    sk_result_id int,
    primary key (sk_community_id, sk_result_id)
);

CREATE TABLE fos (
    sk_id int primary key,
    label varchar(1000)
); -- fos is field of science

CREATE TABLE result_fos (
    sk_fos_id int,
    sk_result_id int,
    primary key (sk_fos_id, sk_result_id)
);

-- result.sk_id can be joined with result_author.sk_result_id
-- result.sk_id can be joined with result_citations.sk_result_id_cited
-- result.sk_id can be joined with result_citations.sk_result_id_cites
-- result.sk_id can be joined with result_community.sk_result_id
-- author.sk_id can be joined with result_author.sk_author_id
-- community.sk_id can be joined with result_community.sk_community_id
-- result.sk_id can be joined with result_fos.sk_result_id
"""


def fc4eosc_database_type() -> str:
    return "Postgres"


def fc4eosc_schema(format: Literal["ddl", "compact"]) -> str:
    """
    Returns the schema of the fc4eosc database in the requested format.
    Args:
        format: THe format of the database schema.

            - str: The schema is returned with the format table1(column1.1, ...), table2(column2.1, ...)
            - ddl: The schema is returned with the CREATE statements used for its creation
    """

    match format:
        case "compact":
            return COMPACT_SCHEMA
        case "ddl":
            return DDL_SCHEMA
