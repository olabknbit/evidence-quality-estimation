# TODO try to make publication type extraction more accurate

import re
from typing import List, Callable

from ebm.reference import Reference

meta_analysis_key = 'meta_analysis'
cohort_study_key = 'cohort_study'
systematic_review_key = 'systematic_review'
other_clinical_trial_key = 'other_clinical_trial'
consensus_key = 'consensus'
randomized_controlled_trial_key = 'randomized_controlled_trial'
practice_guidelines_key = 'practice_guidelines'
review_key = 'review'
case_report_key = 'case_report'
other_study_key = 'other_study'
unknown_key = 'unknown'
cross_sectional_key = "cross_sectional"
expert_opinion_key = "expert_opinion"
evaluation_study_key = "evaluation_study"
multi_center_study_key = "multi_center_study"
comparative_study_key = "comparative_study"
low_quality_study_key = "low_quality_study"


def __check_regexes(regexes: List[str], text: str) -> bool:
    for regex in regexes:
        if re.search(regex, text):
            return True
    return False


########################################################################################################################
# check for common features
########################################################################################################################


def is_randomised(text: str) -> bool:
    regexes_a = ['random']

    regexes_b = \
        ['alloc',
         'chose'
         'assign',
         'appli',
         'desig',
         'animal',
         'patien',
         'subjec',
         'group']

    regexes_c = ['parallel[\W]*group']
    return ((__check_regexes(regexes_a, text) and __check_regexes(regexes_b, text))
            or __check_regexes(regexes_c, text)) \
           and not is_not_randomised(text)


def is_not_randomised(text: str) -> bool:
    regexes = \
        ['coin\W*flip',
         'non\W*random',
         'odd\W*even',
         'uncontrolled\W*study']
    return __check_regexes(regexes, text)
    # return False


def is_blinded(text: str) -> bool:
    regexes = ['doubl.*blind']
    return __check_regexes(regexes, text)


def is_controlled_trial(text: str) -> bool:
    regexes_b = \
        ['control.*stud',
         'clinic.*stud',
         'control.*trial',
         'clinic.*trial'
         ]

    return __check_regexes(regexes_b, text) and not 'uncontrol' in text


def is_clinical_trial(text: str) -> bool:
    regexes_b = \
        [
            'clinic.*stud',
            'clinic.*trial'
        ]

    return __check_regexes(regexes_b, text) and not 'uncontrol' in text


########################################################################################################################
# A mostly
########################################################################################################################
def is_meta_analysis(text: str) -> bool:
    regexes = [
        'meta[-]analys',
        # 'meta[-]synthes',
        # 'meta.*stud'
    ]
    return __check_regexes(regexes, text)


def is_systematic_review(text: str) -> bool:
    regexes = [
        'cochr.*medlin',
        'cochr.*collab',
        'search.*cochr',
        'search.*embas',
        'search.*medlin.*datab',
        'cinahl.*search',
        'literat.*embas',
        'medic.*liter.*rand',
        'narrat.*review',
        'systematic.*review',
        'cochr',
        # 'systematic'
    ]
    return __check_regexes(regexes, text)


def is_randomized_controlled_trial_alt(text: str) -> bool:
    if not is_randomised(text):
        return False

    if is_controlled_trial(text) or is_blinded(text) or is_clinical_trial(text):
        return True

    if 'RCT' in text:
        return True

    return False


# Not used
def is_randomized_controlled_trial(text: str) -> bool:
    regexes = \
        ['random.*alloc',
         'random.*chose'
         'chose.*random',
         'random.*assign',
         'assign.*random',
         'random.*appli',
         'appli.*random',
         'desig:.*random',
         'animal.random',
         'random.*animal',
         'patien.*random',
         'subjec.*random',
         'randomi[sz].*group',
         'parallel[\W]*group',
         'group.*random',
         'random.*doub.*blind',
         'random.*open[\W]*label',
         'randomi[sz]e.*trial',
         'doubl.*blin']
    return __check_regexes(regexes, text) and not is_not_randomised(text)


########################################################################################################################
# B mostly
########################################################################################################################
def is_cohort_study_alt(text: str) -> bool:
    regexes_prospect_or_retrospect = \
        ['retrospect',
         'prospect',
         'observat']
    regexes_study = \
        ['stud',
         'foll',
         'analys',
         'surv',
         'trial'
         ]
    regexes_a = \
        ['forward.*look',
         'backward.*look',
         'patien.*foll.*year',
         'cohort']

    return (__check_regexes(regexes_prospect_or_retrospect, text) and __check_regexes(regexes_study, text)) \
           or __check_regexes(regexes_a, text)


def is_cohort_study(text: str) -> bool:
    regexes = [
        'retrospect.*stud',
        'foll.*prospect',
        'cohort.*stud',
        'retrospect.analys',
        'patien.*foll.*year',
        'foll.*cohort',
        'retrospect.*trial',
        'prospect.*analys',
    ]
    return __check_regexes(regexes, text)


# B is not randomised
def is_other_clinical_trial(text: str) -> bool:
    return is_controlled_trial(text) or is_blinded(text)


# or cross-over
def is_crosssectional_study(text: str) -> bool:
    regexes = \
        ['cross.*stud',
         'long.*stud',
         'cross.*section']
    return __check_regexes(regexes, text)


def is_other_study(text: str) -> bool:
    regexes = \
        ['compar.*group',
         'walk.*in',
         'stud',
         'observ']
    return __check_regexes(regexes, text)


########################################################################################################################
# C mostly
########################################################################################################################
def is_review(text: str) -> bool:
    regexes = \
        [
            # 'review',
            # 'comparativ.*stud',
            'postmar.*survei.*surv',
            'retrospec.*char.*rev',
            'review']
    return __check_regexes(regexes, text)


def is_case_series(text: str) -> bool:
    regexes = \
        ['case.*report',
         'case.*stud',
         # 'journal.*article',
         'case.*series']
    return __check_regexes(regexes, text)


def is_practice_guideline(text: str) -> bool:
    regexes = \
        [
            'guideline',
            'practi.*guidelin',
            'guidel.*diag.*treat',
            'clinic.*guidel']
    return __check_regexes(regexes, text)


# and expert opinion
def is_opinion(text: str) -> bool:
    regexes = \
        ['recommend',
         'opinion']
    return __check_regexes(regexes, text)


def is_consensus_development_conference(text: str) -> bool:
    regexes = \
        ['consens.*confer',
         'consens.*statem',
         'consens.*devel',
         'reac.*consens',
         # 'editorial',
         # 'comment',
         # 'bench.*research',
         # 'practic'
         ]
    return __check_regexes(regexes, text)


def is_evaluation_study(text: str) -> bool:
    regexes = ['eval.*stud']
    return __check_regexes(regexes, text)


def is_multi_center_study(text: str) -> bool:
    regexes = ['multicenter.*stud']
    return __check_regexes(regexes, text)


def is_comparative_study(text: str) -> bool:
    regexes = ['comparativ.*stud']
    return __check_regexes(regexes, text)


########################################################################################################################
# end
########################################################################################################################


def extract_from_text(text: str) -> List[str]:
    text = text.lower()

    matched_regexes = []

    # Meta-analysis
    if is_meta_analysis(text):
        matched_regexes.append(meta_analysis_key)
    # Systematic Review
    if is_systematic_review(text):
        matched_regexes.append(systematic_review_key)
    # Consensus Development Conference
    if is_consensus_development_conference(text):
        matched_regexes.append(consensus_key)
    # Review
    # if is_review(text):
    #     matched_regexes.append(review_key)
    #  Practice Guideline
    if is_practice_guideline(text):
        matched_regexes.append(practice_guidelines_key)
    #  RTC
    if is_randomized_controlled_trial(text):
        matched_regexes.append(randomized_controlled_trial_key)
    # Other Clinical Trial
    if is_other_clinical_trial(text):
        matched_regexes.append(other_clinical_trial_key)
    # Cohort Study
    if is_cohort_study(text):
        matched_regexes.append(cohort_study_key)
    # Case Series
    if is_case_series(text):
        matched_regexes.append(case_report_key)
    # Cross-sectional study
    if is_crosssectional_study(text):
        matched_regexes.append(cross_sectional_key)
    # Opinion
    if is_opinion(text):
        matched_regexes.append(expert_opinion_key)
    if is_evaluation_study(text):
        matched_regexes.append(evaluation_study_key)
    if is_multi_center_study(text):
        matched_regexes.append(multi_center_study_key)
    # Other Study
    if is_other_study(text):
        matched_regexes.append(other_study_key)

    return matched_regexes


def new_method_alt(reference: Reference) -> str:
    pub_types = extract_from_text(
        ' '.join(
            [
                ' '.join(reference.publication_types),
                ' '.join(reference.mesh_headings_majortopic_y),
                ' '.join(reference.mesh_headings_majortopic_n),
                reference.to_line(),
            ]
        )
    )

    # print(reference.id, '\t', pub_types)
    pub_type = pub_types[0] if len(pub_types) > 0 else "unknown"

    return pub_type


def new_method2(reference: Reference) -> str:
    pub_types = extract_from_text(' '.join(reference.publication_types))
    if not pub_types:
        pub_types = extract_from_text(' '.join(
            [
                ' '.join(reference.mesh_headings_majortopic_y),
                ' '.join(reference.mesh_headings_majortopic_n),
            ]))
    if not pub_types:
        pub_types = extract_from_text(reference.title)
    if not pub_types:
        sentences = reference.abstract.split('. ')
        for sentence in sentences:
            pub_types = extract_from_text(sentence)
            if pub_types:
                break
    # print(reference.id, '\t', pub_types)
    pub_type = pub_types[0] if len(pub_types) > 0 else "unknown"

    return pub_type


def new_method(reference: Reference) -> str:
    publication_types_metadata = ' '.join(reference.publication_types).lower()
    title = reference.title.lower()
    abstract = reference.abstract.lower()

    def check_everything(f: Callable) -> bool:
        everything = [publication_types_metadata, title]
        everything.extend([sentence for sentence in abstract.split('. ')])
        return any([f(el) for el in everything])

    def check_regexes_only(f: Callable) -> bool:
        everything = [title]
        everything.extend([sentence for sentence in abstract.split('. ')])
        return any([f(el) for el in everything])

    # Meta-analysis or Systematic Review
    if check_everything(is_meta_analysis) or check_regexes_only(is_systematic_review):
        return meta_analysis_key

    # Consensus Development Conference
    if check_everything(is_consensus_development_conference):
        consensus_development_conference_key = low_quality_study_key
        return consensus_development_conference_key
    # Review
    if check_everything(is_review):
        return review_key
    #  RTC
    if check_everything(is_randomized_controlled_trial):
        return randomized_controlled_trial_key
    # Other Clinical Trial
    if is_other_clinical_trial(publication_types_metadata):
        return other_clinical_trial_key

    # Cohort Study
    if check_regexes_only(is_cohort_study):
        return cohort_study_key

    #  Practice Guideline
    if check_everything(is_practice_guideline):
        return practice_guidelines_key
    # Case Series
    if check_everything(is_case_series):
        return case_report_key
    # Cross-sectional study
    if check_everything(is_crosssectional_study):
        cross_sectional_key = other_study_key
        return cross_sectional_key
    # Opinion
    if check_everything(is_opinion):
        expert_opinion_key = low_quality_study_key
        return expert_opinion_key
    # Evaluation study
    if check_everything(is_evaluation_study):
        evaluation_study_key = low_quality_study_key
        return evaluation_study_key
    # Multi center study
    if check_everything(is_multi_center_study):
        multi_center_study_key = other_study_key
        return multi_center_study_key
    if check_everything(is_comparative_study):
        # Comparative Studies
        return comparative_study_key
    # Other Study
    if check_everything(is_other_study):
        return other_study_key
    return unknown_key
