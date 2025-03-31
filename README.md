# FC4EOSC NL Search Service

The NL Search service allows users to search the RDGraph using Natural Language (NL).
This allows non-technical users to pose complex questions without the need for expertise in query languages or any other technology.
The system takes care of translating the user's NL question into a SQL query, that is then executed on the RDGraph database and results are seemlessly returned.


The API documentation can be found here: [API Documentation](https://darelab.athenarc.gr/nl_search/docs).

This repo includes the source code under `darelabdb/` and the wheel file (`api_nl_search-0.1.0-py3-none-any.whl`) for the NL Search service.

## Installation

```bash
git clone git@github.com:athenarc/fc4eosc-nlsearch.git
cd fc4eosc-nlsearch
pip install api_nl_search-0.1.0-py3-none-any.whl
```

Then you will have to create a `.env` file in the root of the project following the `.env.example` file.
