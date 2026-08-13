# -*- coding: utf-8 -*-
"""Parse Orphanet disease alignment JSON/XML dumps."""

import json
from pathlib import Path

import pandas as pd
import xmltodict

ROOT = Path(__file__).resolve().parent


def disease_gene():
    file = ROOT / "gene_disease.xml"
    datas = xmltodict.parse(
        open(file, "r", encoding="ISO-8859-1").read()
    )["JDBOR"]["DisorderList"]["Disorder"]
    for data in datas:
        print(data)
        symbol = data["DisorderGeneAssociationList"]["DisorderGeneAssociation"][
            "Gene"
        ]["Symbol"]
        disease_code = data["OrphaCode"]


def disease_en():
    file = ROOT / "disease_alignment_en.json"
    datas = json.load(open(file, "r", encoding="utf-8"))["JDBOR"][0][
        "DisorderList"
    ][0]["Disorder"]
    res = []
    for data in datas:
        id = data["id"]
        orphaCode = data["OrphaCode"]
        disease_name = data["Name"][0]["label"]
        externalReferenceList = data["ExternalReferenceList"][0]
        external_id_ls = set()
        if int(externalReferenceList["count"]) > 0:
            for i in externalReferenceList["ExternalReference"]:
                external_id_ls.add((i["Source"], i["Reference"]))
        synonym = data["SynonymList"][0]
        synonym_ls = set()
        if int(synonym["count"]) > 0:
            for i in synonym["Synonym"]:
                synonym_ls.add(i["label"])

        summaryList = data["SummaryInformationList"][0]
        summary_ls = set()
        if int(summaryList["count"]) > 0:
            for i in summaryList["SummaryInformation"]:
                TextSectionList = i["TextSectionList"][0]
                if int(TextSectionList["count"]) > 0:
                    label = TextSectionList["TextSection"][0]["TextSectionType"][0][
                        "Name"
                    ][0]["label"]
                    content = TextSectionList["TextSection"][0]["Contents"]
                    summary_ls.add((label, content))
        external_id_ls = "|".join(i[0] + ":" + i[1] for i in external_id_ls)
        synonym_ls = "|".join(synonym_ls)
        # At most one definition text section per entity in practice.
        summary_ls = "|".join(i[0] + ":" + i[1] for i in summary_ls)
        res.append(
            [id, orphaCode, disease_name, external_id_ls, synonym_ls, summary_ls]
        )
    pd.DataFrame(
        res,
        columns=[
            "id",
            "orphaCode",
            "disease_name",
            "external_id_ls",
            "synonym_ls",
            "summary_ls",
        ],
    ).to_csv(ROOT / "disease_en.csv", index=False)


def disease_zh():
    file = ROOT / "disease_alignment_zh.json"
    datas = json.load(open(file, "r", encoding="utf-8"))["JDBOR"][0][
        "DisorderList"
    ][0]["Disorder"]
    res = []
    for data in datas:
        id = data["id"]
        orphaCode = data["OrphaCode"]
        disease_name = data["Name"][0]["label"]
        externalReferenceList = data["ExternalReferenceList"][0]
        external_id_ls = set()
        if int(externalReferenceList["count"]) > 0:
            for i in externalReferenceList["ExternalReference"]:
                external_id_ls.add((i["Source"], i["Reference"]))
        synonym = data["SynonymList"][0]
        synonym_ls = set()
        if int(synonym["count"]) > 0:
            for i in synonym["Synonym"]:
                if i["lang"] == "zh":
                    synonym_ls.add(i["label"])

        summaryList = data["SummaryInformationList"][0]
        summary_ls = set()
        if int(summaryList["count"]) > 0:
            for i in summaryList["SummaryInformation"]:
                TextSectionList = i["TextSectionList"][0]
                if int(TextSectionList["count"]) > 0:
                    label = TextSectionList["TextSection"][0]["TextSectionType"][0][
                        "Name"
                    ][0]["label"]
                    content = TextSectionList["TextSection"][0]["Contents"]
                    summary_ls.add((label, content))
        external_id_ls = "|".join(i[0] + ":" + i[1] for i in external_id_ls)
        synonym_ls = "|".join(synonym_ls)
        # At most one definition text section per entity in practice.
        summary_ls = "|".join(i[0] + ":" + i[1] for i in summary_ls)
        res.append(
            [id, orphaCode, disease_name, external_id_ls, synonym_ls, summary_ls]
        )
    pd.DataFrame(
        res,
        columns=[
            "id",
            "orphaCode",
            "disease_name",
            "external_id_ls",
            "synonym_ls",
            "summary_ls",
        ],
    ).to_csv(ROOT / "disease_zh.csv", index=False)


if __name__ == "__main__":
    disease_zh()
