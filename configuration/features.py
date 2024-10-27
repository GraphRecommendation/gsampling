from shared.configuration_classes import FeatureConfiguration
from shared.enums import FeatureEnum

graphsage = FeatureConfiguration('graphsage', 'graphsage', 'ckg', True, FeatureEnum.ENTITIES,
                                      attributes=['name', 'description'])

comsage = FeatureConfiguration('comsage', None, 'kg', True, FeatureEnum.ENTITIES, scale=True)
transsage = FeatureConfiguration('transsage', None, 'kg', True, FeatureEnum.ENTITIES, scale=True)
combined_methods = [comsage, transsage]


transr = FeatureConfiguration('transr_300', 'complex', 'kg', False, FeatureEnum.DESC_ENTITIES, model_name='transr',
                              embedding_dim=300)
complEX = FeatureConfiguration('complex', 'complex', 'kg', False, FeatureEnum.DESC_ENTITIES, model_name='complex')
transe = FeatureConfiguration('transe', 'complex', 'kg', False, FeatureEnum.DESC_ENTITIES, model_name='transe')
other_methods = [transr, complEX, transe]

# Feature definitions
feature_configurations = [graphsage]  + combined_methods + other_methods
feature_conf_names = [f.name for f in feature_configurations]