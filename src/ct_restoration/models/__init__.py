"""Learned restoration models.

Two architectures live here, and one shared contract they both satisfy:

* :mod:`~ct_restoration.models.base` - what the benchmark, the adapter and
  the training loop are allowed to assume about any learned method;
* :mod:`~ct_restoration.models.cnn` - the Milestone 8 residual CNN, 28,353
  parameters, an 11x11 receptive field;
* :mod:`~ct_restoration.models.unet` - the Milestone 9 lightweight residual
  U-Net, 116,753 parameters, a 44x44 maximum deepest-path receptive field;
* :mod:`~ct_restoration.models.adapter` - the bridge into the benchmark's
  one-argument restoration contract, written against the shared protocol so
  neither architecture gets its own evaluation path;
* :mod:`~ct_restoration.models.diagnostics` - one definition of the raw and
  post-clamp correction statistics, shared by both.

Nothing here is re-exported from :mod:`ct_restoration.data` or anywhere else:
a model is a method under test, not part of the benchmark definition, and the
evaluation path must stay importable without one. Nothing here is imported at
package level either, so importing the benchmark does not pull in torch.
"""
