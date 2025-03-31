from typing import Literal

COMPACT_SCHEMA = """\
activity_types(code, description),
countries(unics_id, name, alpha_2, alpha_3, nuts_0),
ec_framework_programs(name),
erc_research_domains(code, description),
erc_panels(code, description, part_of),
eu_territorial_units(nuts_code, description, nuts_level, nuts_version),
funding_schemes(code, title),
institutions(unics_id, country_id, name, nuts3_code, db_pedia_url, wikidata_url, grid_id, acronym, short_name, website),
people(unics_id, full_name),
programmes(code, rcn, title, short_name, parent),
project_member_roles(code, description),
projects(unics_id, acronym, title, ec_call, ec_fund_scheme, cordis_ref, ec_ref, start_date, end_date, start_year, end_year, homepage, total_cost, ec_max_contribution, framework_program, objective, principal_investigator),
project_erc_panels(project, panel),
project_members(unics_id, project, pic_number, rcn, member_name, activity_type, country, street, city, postal_code, ec_contribution, institution_id, member_role, nuts3_code, member_short_name, department_name, vat_number, latitude, longitude),
project_programmes(project, programme),
subject_areas(code, title, description),
project_subject_areas(project, subject_area),
topics(code, rcn, title),
project_topics(project, topic)
"""

DDL_SCHEMA = """\
create table
    activity_types (
        code varchar(255) primary key,
        description longtext
    )

    create table
    countries (
        unics_id int primary key,
        name longtext,
        alpha_2 varchar(2),
        alpha_3 varchar(3),
        nuts_0 varchar(2)
    ) create fulltext index ts_idx_countries_name on countries (name);

create table
    ec_framework_programs (name varchar(255) primary key) create fulltext index ts_idx_ec_framework_programs_name on ec_framework_programs (name);

create table
    erc_research_domains (
        code varchar(255) primary key,
        description longtext
    )
create table
    erc_panels (
        code varchar(255) primary key,
        description longtext,
        part_of varchar(200),
        constraint erc_panels_part_of_fkey foreign key (part_of) references erc_research_domains (code)
    ) create fulltext index ts_idx_erc_panels_description on erc_panels (description);

create table
    eu_territorial_units (
        nuts_code varchar(255) primary key,
        description longtext,
        nuts_level int,
        nuts_version longtext
    ) create fulltext index ts_idx_eu_territorial_units_description on eu_territorial_units (description);

create table
    funding_schemes (
        code varchar(255) primary key,
        title longtext
    )

    create table
    institutions (
        unics_id int primary key,
        country_id int,
        name longtext,
        nuts3_code varchar(200),
        db_pedia_url longtext,
        wikidata_url longtext,
        grid_id longtext,
        acronym longtext,
        short_name longtext,
        website longtext,
        constraint institutions_country_id_fkey foreign key (country_id) references countries (unics_id),
        constraint institutions_nuts3_code_fkey foreign key (nuts3_code) references eu_territorial_units (nuts_code)
    ) create fulltext index ts_idx_institutions_name on institutions (name);

create table
    people (unics_id int primary key, full_name longtext)

create table
    programmes (
        code varchar(255) primary key,
        rcn longtext,
        title longtext,
        short_name longtext,
        parent varchar(200),
        constraint programmes_parent_fkey foreign key (parent) references programmes (code)
    ) create fulltext index ts_idx_programmes_title on programmes (title);

create table
    project_member_roles (
        code varchar(255) primary key,
        description longtext
    )
create table
    projects (
        unics_id int primary key,
        acronym longtext,
        title longtext,
        ec_call longtext,
        ec_fund_scheme varchar(255),
        cordis_ref longtext,
        ec_ref longtext,
        start_date date,
        end_date date,
        start_year int,
        end_year int,
        homepage longtext,
        total_cost double,
        ec_max_contribution double,
        framework_program varchar(255),
        objective longtext,
        principal_investigator int,
        constraint projects_cordis_ref_key unique (cordis_ref (255)),
        constraint projects_ec_ref_key unique (ec_ref (255)),
        constraint projects_ec_fund_scheme_fkey foreign key (ec_fund_scheme) references funding_schemes (code),
        constraint projects_framework_program_fkey foreign key (framework_program) references ec_framework_programs (name),
        constraint projects_principal_investigator_fkey foreign key (principal_investigator) references people (unics_id)
    )

create table
    project_erc_panels (
        project int primary key,
        panel varchar(255),
        constraint project_erc_panels_panel_fkey foreign key (panel) references erc_panels (code),
        constraint project_erc_panels_project_fkey foreign key (project) references projects (unics_id)
    )

create table
    project_members (
        unics_id int primary key,
        project int,
        pic_number longtext,
        rcn longtext,
        member_name longtext,
        activity_type varchar(255),
        country longtext,
        street longtext,
        city longtext,
        postal_code longtext,
        ec_contribution double,
        institution_id int,
        member_role varchar(255),
        nuts3_code varchar(255),
        member_short_name longtext,
        department_name longtext,
        vat_number longtext,
        latitude decimal,
        longitude decimal,
        constraint project_members_activity_type_fkey foreign key (activity_type) references activity_types (code),
        constraint project_members_institution_id_fkey foreign key (institution_id) references institutions (unics_id),
        constraint project_members_member_role_fkey foreign key (member_role) references project_member_roles (code),
        constraint project_members_nuts3_code_fkey foreign key (nuts3_code) references eu_territorial_units (nuts_code),
        constraint project_members_project_fkey foreign key (project) references projects (unics_id)
    ) create index project_members_institution_id_idx on project_members (institution_id);

create fulltext index ts_idx_project_members_city on project_members (city);

create fulltext index ts_idx_project_members_member_name on project_members (member_name);

create fulltext index ts_idx_project_members_member_role on project_members (member_role);

create fulltext index ts_idx_project_members_member_short_name on project_members (member_short_name);

create fulltext index ts_idx_project_members_street on project_members (street);

create table
    project_programmes (
        project int,
        programme varchar(255),
        primary key (project, programme),
        constraint project_programmes_programme_fkey foreign key (programme) references programmes (code),
        constraint project_programmes_project_fkey foreign key (project) references projects (unics_id)
    ) create fulltext index ts_idx_projects_acronym on projects (acronym);

create fulltext index ts_idx_projects_title on projects (title);

create table
    subject_areas (
        code varchar(255) primary key,
        title longtext,
        description longtext
    )

create table
    project_subject_areas (
        project int,
        subject_area varchar(255),
        primary key (project, subject_area),
        constraint project_subject_areas_project_fkey foreign key (project) references projects (unics_id),
        constraint project_subject_areas_subject_area_fkey foreign key (subject_area) references subject_areas (code)
    ) create fulltext index ts_idx_subject_areas_title on subject_areas (title);

create table
    topics (
        code varchar(255) primary key,
        rcn longtext,
        title longtext
    )

create table
    project_topics (
        project int,
        topic varchar(255),
        primary key (project, topic),
        constraint project_topics_project_fkey foreign key (project) references projects (unics_id),
        constraint project_topics_topic_fkey foreign key (topic) references topics (code)
    ) create fulltext index ts_idx_topics_title on topics (title);
            """


def cordis_database_type() -> str:
    return "MySQL"


def cordis_schema(format: Literal["ddl", "compact"]) -> str:
    """
    Returns the schema of the cordis database in the requested format.
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
