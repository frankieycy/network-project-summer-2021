## Neuronal Network Modeling

* Folder 0 contains my progress notes of all trials and errors throughout the entire project (so it can be slightly disorganized).

* Folder 1 contains the reports and presentation slides.

* Folders 2-6 contain codes relevant to network models and reconstruction techniques.

    - Each folder is a self-contained collection of codes, including libraries and/or example files. The example files represent minimal working examples which can be run directly, for example, in the terminal via: `python Example-file-name.py`. The examples by no means demonstrate all functionalities of the libraries but only the core ones.

    - They are all written in Python, and the packages required include: *numpy, scipy, sklearn, matplotlib, pycorrelate, tqdm*. For the specific instructions, refer to README inside the folders.

* Folder 7 contains coupling strength and spike timestamp data used over the project.

* Folder 8 contains some results including:
    - FNCCH Reconstructions - reconstructed coupling strength matrices and intermediate calculations (e.g. tau*)
    - DIV66 Reference Networks - coupling strengths and distributional stats
    - DIV66 Spk Neuron Model Ref Network Simulations (500ms) - timestamps and corresponding raster plots
    - DIV25 Spk Neuron Model on Different dt - convergence of dynamics with respect to dt
    - (Chris & Samuel's) Spk Neuron Model Orig Network Simulations (7500 & 15000ms) - simulation results over 7500 & 15000ms
