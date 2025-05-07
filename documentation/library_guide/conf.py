#===============================================================================
# Copyright Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'oneDPL Documentation'
copyright = 'Intel Corporation'
author = 'Intel'

# The full version, including alpha/beta/rc tags
release = '2022.9.0'

rst_epilog = """
.. include:: /variables.txt
.. include:: /common_cross_document_links.txt
"""

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx-prompt',
    'sphinx_substitution_extensions'
	]

docbundle_settings = {
    'csv_dir':'csv_dir'
}

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_logo = '_static/oneAPI-rgb-rev-100.png'
html_favicon = '_static/favicons.png'
#latex_logo = '_static/oneAPI-rgb-3000.png'
html_show_sourcelink = False
#html_js_files = ['custom.js']

# html_context = {
#     'css_files': [
#         '_static/style.css',  # override wide tables in RTD theme
#         ],
#     }

# html_theme = 'otc_tcs_sphinx_theme'
# html_theme_path = ['_themes']

# import sphinx_rtd_theme
html_theme = 'sphinx_book_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
#if on_rtd:
#    using_rtd_theme = True

# Theme options
html_theme_options = {
    # 'typekit_id': 'hiw1hhg',
    # 'analytics_id': '',
    # 'sticky_navigation': True,  # Set to False to disable the sticky nav while scrolling.
    #'logo_only': False,  # if we have a html_logo below, this shows /only/ the logo with no title text
    #'collapse_navigation': False,  # Collapse navigation (False makes it tree-like)
    #'navigation_depth': 4  # Depth of the headers shown in the navigation bar
    #'display_version': True,  # Display the docs version
    'repository_url': 'https://github.com/uxlfoundation/oneDPL',
    'path_to_docs': 'documentation/library_guide',
    'use_issues_button': True,
    'use_edit_page_button': True,
    'repository_branch': 'main',
    'extra_footer': '<p align="right"><a data-cookie-notice="true" href="https://www.intel.com/content/www/us/en/privacy/intel-cookie-notice.html">Cookies</a></p>'
}
