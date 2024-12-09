_base_ = ["./meta_test_base.py"]

#VeLO uses defaults and the following hyperparameters:
# opt_from_checkpoint__6cf1d6ba_d295_4f96_88f3_ca14cdaf0da9/HyperV2.lstm_hidden_size = 512
# opt_from_checkpoint__6cf1d6ba_d295_4f96_88f3_ca14cdaf0da9/HyperV2.param_inits = 256
# opt_from_checkpoint__6cf1d6ba_d295_4f96_88f3_ca14cdaf0da9/HyperV2.use_bugged_loss_features = False

hyper_v2_args =  dict(
      interp_mult=1.0,
      lstm_hidden_size=16,
      param_inits=8,
      ff_hidden_size=4,
      ff_hidden_layers=2,
      initial_momentum_decays=(0.9, 0.99, 0.999),
      initial_rms_decays=(0.999,),
      initial_adafactor_decays=(0.9, 0.99, 0.999),
      mix_layers=True,
      exp_mult=0.001,
      step_mult=0.001,
      validation_mode=False,
      with_validation_feature_dim=False,

      # ablation flags.
      identity_lstm_controls=False,
      identity_lr_mult=False,

      with_g=True,
      with_m=True,
      with_m_feat=True,
      with_rms=True,
      with_rms_feat=True,
      with_rms_norm_g=True,
      with_rsqrt_rms=True,
      with_p=True,
      with_fac_norm_g=True,
      with_fac_rms=True,
      with_fac_rsqrt=True,
      with_grad_clip_feat=True,
      with_fac_mom_mult=True,
      with_rms_only_norm_g=True,
      adafactor_accumulator=True,

      param_scale_mult=False,
      
      use_bugged_next_lstm_state=False,
      use_bugged_loss_features=False,
      precondition_output=False,
      reparam_decay=10.,
      rnn_state_decay=0.0,

      # more summaries
      summarize_each_layer=False,
      summarize_all_control=False,

      # Modify the lopt to probe behavior
      constant_loss=False,
      clip_param_scale_amount=None,)