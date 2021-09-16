from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = '/gdrive/My Drive/DAVIS/'
    settings.got10k_path = '/gdrive/My Drive'
    settings.got_packed_results_path = '/gdrive/My Drive'
    settings.got_reports_path = '/gdrive/My Drive'
    settings.lasot_path = '/gdrive/My Drive'
    settings.network_path = '/gdrive/My Drive/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = '/gdrive/My Drive'
    settings.otb_path = '/gdrive/My Drive'
    settings.result_plot_path = '/gdrive/My Drive/pytracking/pytracking/result_plots/'
    settings.results_path = '/gdrive/My Drive/pytracking/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/gdrive/My Drive/pytracking/pytracking/segmentation_results/'
    settings.tn_packed_results_path = '/gdrive/My Drive'
    settings.tpl_path = '/gdrive/My Drive'
    settings.trackingnet_path = '/gdrive/My Drive'
    settings.uav_path = '/gdrive/My Drive'
    settings.vot_path = '/gdrive/My Drive'
    settings.youtubevos_dir = '/gdrive/My Drive'
    settings.mot_dir ='/gdrive/My Drive'
 
    return settings