from shared.configuration_classes import DatasetConfiguration, CountFilter
from shared.enums import Sentiment

# region Dataset relations
"""
Possible amazon-book relations 
['http://rdf.freebase.com/ns/type.object.type', 'http://rdf.freebase.com/ns/type.type.instance', 'http://rdf.freebase.com/ns/book.written_work.copyright_date', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://rdf.freebase.com/ns/kg.object_profile.prominent_type', 'http://rdf.freebase.com/ns/book.written_work.subjects', 'http://rdf.freebase.com/ns/book.written_work.date_of_first_publication', 'http://rdf.freebase.com/ns/common.topic.notable_types', 'http://rdf.freebase.com/ns/book.book_subject.works', 'http://rdf.freebase.com/ns/media_common.literary_genre.books_in_this_genre', 'http://rdf.freebase.com/ns/book.written_work.author', 'http://rdf.freebase.com/ns/book.written_work.original_language', 'http://rdf.freebase.com/ns/freebase.valuenotation.is_reviewed', 'http://rdf.freebase.com/ns/book.book.genre', 'http://rdf.freebase.com/ns/book.author.works_written', 'http://rdf.freebase.com/ns/book.written_work.previous_in_series', 'http://rdf.freebase.com/ns/book.literary_series.works_in_this_series', 'http://rdf.freebase.com/ns/book.book_character.appears_in_book', 'http://rdf.freebase.com/ns/book.written_work.part_of_series', 'http://rdf.freebase.com/ns/book.book.characters', 'http://rdf.freebase.com/ns/book.written_work.next_in_series', 'http://www.w3.org/2000/01/rdf-schema#label', 'http://rdf.freebase.com/ns/freebase.valuenotation.has_value', 'http://rdf.freebase.com/ns/theater.play.country_of_origin', 'http://rdf.freebase.com/ns/book.short_story.genre', 'http://rdf.freebase.com/ns/fictional_universe.work_of_fiction.part_of_these_fictional_universes', 'http://rdf.freebase.com/ns/theater.play.genre', 'http://rdf.freebase.com/ns/book.illustrator.books_illustrated', 'http://rdf.freebase.com/ns/media_common.literary_genre.stories_in_this_genre', 'http://rdf.freebase.com/ns/type.object.name', 'http://rdf.freebase.com/ns/fictional_universe.fictional_universe.works_set_here', 'http://rdf.freebase.com/ns/common.topic.topical_webpage', 'http://rdf.freebase.com/ns/theater.play.date_of_first_performance', 'http://rdf.freebase.com/ns/freebase.valuenotation.has_no_value', 'http://rdf.freebase.com/ns/theater.theater_genre.plays_in_this_genre', 'http://rdf.freebase.com/ns/common.topic.official_website', 'http://rdf.freebase.com/ns/book.book.interior_illustrations_by', 'http://rdf.freebase.com/ns/book.written_work.date_written', 'http://rdf.freebase.com/ns/base.yupgrade.user.topics']
"""
ab_relations = ['http://rdf.freebase.com/ns/type.object.type','http://www.w3.org/1999/02/22-rdf-syntax-ns#type']
"""
Possible yelp relations
['http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'https://purl.archive.org/purl/yckg/vocabulary#businessProperty', 'https://purl.archive.org/purl/yckg/vocabulary#hasAmbience', 'https://purl.archive.org/purl/yckg/vocabulary#hasBestNights', 'https://purl.archive.org/purl/yckg/vocabulary#hasBusinessParking', 'https://purl.archive.org/purl/yckg/vocabulary#hasDietaryRestrictions', 'https://purl.archive.org/purl/yckg/vocabulary#hasGoodForMeal', 'https://purl.archive.org/purl/yckg/vocabulary#hasHairSpecializesIn', 'https://purl.archive.org/purl/yckg/vocabulary#hasMusic', 'https://schema.org/keywords', 'https://schema.org/location', 'https://www.wikidata.org/wiki/Property:P131']
"""
yelp_relations = [
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'https://purl.archive.org/purl/yckg/vocabulary#businessProperty',
        'https://purl.archive.org/purl/yckg/vocabulary#hasAmbience',
        'https://purl.archive.org/purl/yckg/vocabulary#hasBestNights',
        'https://purl.archive.org/purl/yckg/vocabulary#hasBusinessParking',
        'https://purl.archive.org/purl/yckg/vocabulary#hasDietaryRestrictions',
        'https://purl.archive.org/purl/yckg/vocabulary#hasGoodForMeal',
        'https://purl.archive.org/purl/yckg/vocabulary#hasHairSpecializesIn',
        'https://purl.archive.org/purl/yckg/vocabulary#hasMusic'
]
"""
Possible ml-mr relations
['DIRECTED_BY', 'FOLLOWED_BY', 'FROM_DECADE', 'HAS_GENRE', 'HAS_SUBJECT', 'PRODUCED_BY', 'STARRING', 'SUBCLASS_OF']
"""
ml_relations = ['SUBCLASS_OF', 'FROM_DECADE']
# endregion

# All users have at least 5 positive ratings, for 5 folds.
# lambda x: {x > 3: 1}.get(True, 0) if above 3 creates a dict with True:1, which we try to return, otherwise 0.
mindreader = DatasetConfiguration('mindreader', lambda x: {x > 0: 1}.get(True, 0),
                                  filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
# region movielens
movielens = DatasetConfiguration('movielens', lambda x: {x < 3: -1, x > 3: 1}.get(True, 0),
                                 filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
ml_mr = DatasetConfiguration('ml-mr', lambda x: {x > 3: 1}.get(True, 0),
                             filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)],
                             time_based_sampling=True, k_core=5)
ml_mr_1m = DatasetConfiguration('ml-mr-1m', lambda x: {x > 3: 1}.get(True, 0),
                                filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)],
                                max_users=None, max_ratings=3000000, time_based_sampling=True, time_based_pruning=True,
                                k_core=5, is_relation_type=ml_relations)

amazon_book = DatasetConfiguration('amazon-book', lambda x: x, k_core=1, is_relation_type=ab_relations)

yelp = DatasetConfiguration('yelpkg', lambda x: {x > 3: 1}.get(True, 0),
                            filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)],
                            k_core=5, time_based_sampling=True, is_relation_type=yelp_relations)


paper_datasets = [ml_mr_1m, ml_mr, amazon_book, yelp]

datasets = [mindreader, movielens, ml_mr, ml_mr_1m, amazon_book, yelp]
dataset_names = [d.name for d in datasets]