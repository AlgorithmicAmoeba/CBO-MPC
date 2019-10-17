.. This is A COPY OF the main index.rst file which is rendered into the landing page of your documentation.
   Follow the inline instructions to configure this for YOUR next project.



Welcome to CBO: MPC's documentation !
=========================================================
|

The code in this project can be used to simulate and tune model predictive controllers
on systems with Laplace transfer functions

The source code is available `here <https://github.com/darren-roos/CBO-MPC>`_.

|

.. maxdepth = 1 means the Table of Contents will only links to the separate pages of the documentation.
   Increasing this number will result in deeper links to subtitles etc.

.. Below is the main Table Of Content
   You have below a "dummy" file, that holds a template for a class.
   To add pages to your documentation:
        * Make a file_name.rst file that follows one of the templates in this project
        * Add its name here to this TOC


.. toctree::
   :maxdepth: 1
   :name: mastertoc

   StepModel
   PlantModel
   ModelPredictiveController

.. raw:: latex

    \chapter{Code}
    \section{Step Model}
    \lstinputlisting{../../../StepModel.py}

    \section{Plant Model}
    \lstinputlisting{../../../PlantModel.py}

    \section{Model Predictive Controller}
    \lstinputlisting{../../../ModelPredictiveController.py}

.. Delete this line until the * to generate index for your project: * :ref:`genindex`

.. Finished personalizing all the relevant details? Great! Now make this your main index.rst,
   And run `make clean html` from your documentation folder :)
