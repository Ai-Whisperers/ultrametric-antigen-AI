from unittest.mock import patch

from src.models.ternary_vae import TernaryVAEV5_11, TernaryVAEV5_11_OptionC
from tests.core.helpers import MockFrozenModule
from tests.factories.models import ModelConfigFactory


class VAEBuilder:
    """Fluent Builder for TernaryVAE instances."""

    def __init__(self):
        self._config = ModelConfigFactory.minimal()
        self._option_c = False
        self._mock_frozen = True
        self._mock_frozen_tuple = False  # whether mocks return tuple
        self._frozen_encoder_mock = None
        self._frozen_decoder_mock = None

    def with_config(self, **kwargs):
        self._config.update(kwargs)
        return self

    def as_option_c(self):
        self._option_c = True
        return self

    def with_dual_projection(self, enabled=True):
        self._config["use_dual_projection"] = enabled
        if enabled:
            self._config["n_projection_layers"] = 2  # Default for dual
        return self

    def with_controller(self, enabled=True):
        self._config["use_controller"] = enabled
        return self

    def with_real_frozen_components(self):
        self._mock_frozen = False
        return self

    def build(self):
        """Builds the model, handling mocking if requested."""
        if self._mock_frozen:
            return self._build_mocked()

        if self._option_c:
            return TernaryVAEV5_11_OptionC(**self._config)
        return TernaryVAEV5_11(**self._config)

    def _build_mocked(self):
        """Builds model with patched frozen components."""
        # This is tricky because constructing the model triggers __init__ which instantiates FrozenEncoder
        # We need to temporarily patch while building

        with patch("src.models.ternary_vae.FrozenEncoder") as mock_enc_cls, patch("src.models.ternary_vae.FrozenDecoder") as mock_dec_cls:

            # Use our MockFrozenModule helper
            def create_enc(*args, **kwargs):
                return MockFrozenModule(
                    output_shape=(self._config["latent_dim"],),
                    return_tuple=True,
                )

            mock_enc_cls.side_effect = create_enc

            def create_dec(*args, **kwargs):
                return MockFrozenModule(output_shape=(9, 3), return_tuple=False)  # Fixed input dim 9

            mock_dec_cls.side_effect = create_dec

            if self._option_c:
                model = TernaryVAEV5_11_OptionC(**self._config)
            else:
                model = TernaryVAEV5_11(**self._config)

            return model
