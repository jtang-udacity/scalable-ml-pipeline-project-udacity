#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum
from pydantic import BaseModel, Field


class WorkclassEnum(str, Enum):
    state = "State-gov"
    selfemp = "Self-emp-not-inc"
    private = "Private"
    federal = "Federal-gov"
    localgov = "Local-gov"
    unk = "?"
    selfempinc = "Self-emp-inc"
    wpay = "Without-pay"
    neverworked = "Never-worked"


class EducationEnum(str, Enum):
    bsc = "Bachelors"
    hs = "HS-grad"
    eleven = "11th"
    msc = "Masters"
    nine = "9th"
    some = "Some-college"
    acdm = "Assoc-acdm"
    voc = "Assoc-voc"
    seven = "7th-8th"
    phd = "Doctorate"
    prof = "Prof-school"
    five = "5th-6th"
    ten = "10th"
    one = "1st-4th"
    pre = "Preschool"
    twelve = "12th"


class MaritalStatusEnum(str, Enum):
    never = "Never-married"
    civ = "Married-civ-spouse"
    div = "Divorced"
    absent = "Married-spouse-absent"
    sep = "Separated"
    af = "Married-AF-spouse"
    widow = "Widowed"


class OccupationEnum(str, Enum):
    adm = "Adm-clerical"
    exe = "Exec-managerial"
    handler = "Handlers-cleaners"
    prof = "Prof-specialty"
    other = "Other-service"
    sale = "Sales"
    crafr = "Craft-repair"
    transport = "Transport-moving"
    farm = "Farming-fishing"
    machine = "Machine-op-inspct"
    tech = "Tech-support"
    unk = "?"
    protectibe = "Protective-serv"
    army = "Armed-Forces"
    priv = "Priv-house-serv"


class RelationshipEnum(str, Enum):
    nofam = "Not-in-family"
    husband = "Husband"
    wife = "Wife"
    ownchild = "Own-child"
    unmar = "Unmarried"
    other = "Other-relative"


class RaceEnum(str, Enum):
    white = "White"
    black = "Black"
    asian = "Asian-Pac-Islander"
    amer = "Amer-Indian-Eskimo"
    other = "Other"


class SexEnum(str, Enum):
    male = "Male"
    female = "Female"


class NativeCountryEnum(str, Enum):
    usa = "United-States"
    cub = "Cuba"
    jam = "Jamaica"
    ind = "India"
    unk = "?"
    mex = "Mexico"
    sou = "South"
    pue = "Puerto-Rico"
    hon = "Honduras"
    uk = "England"
    ca = "Canada"
    de = "Germany"
    ira = "Iran"
    phi = "Philippines"
    it = "Italy"
    po = "Poland"
    col = "Columbia"
    cam = "Cambodia"
    tha = "Thailand"
    ecu = "Ecuador"
    loa = "Laos"
    taiw = "Taiwan"
    hai = "Haiti"
    pt = "Portugal"
    dom = "Dominican-Republic"
    sal = "El-Salvador"
    fr = "France"
    gu = "Guatemala"
    chi = "China"
    jp = "Japan"
    yug = "Yugoslavia"
    per = "Peru"
    out = "Outlying-US(Guam-USVI-etc)"
    scot = "Scotland"
    tt = "Trinadad&Tobago"
    gr = "Greece"
    nic = "Nicaragua"
    viet = "Vietnam"
    hk = "Hong"
    eir = "Ireland"
    hun = "Hungary"
    holand = "Holand-Netherlands"


class PredictPayload(BaseModel):
    age: int
    fnlgt: int
    education_num: int = Field(..., alias="education-num")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    
    workclass: WorkclassEnum
    education: EducationEnum
    marital_status: MaritalStatusEnum = Field(alias="marital-status")
    occupation: OccupationEnum
    relationship: RelationshipEnum
    race: RaceEnum
    sex: SexEnum
    native_country: NativeCountryEnum = Field(alias="native-country")