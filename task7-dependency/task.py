# -*- coding: utf-8 -*-
# https://web.stanford.edu/class/cs224n/
# https://aclanthology.org/W04-0308.pdf
import hanlp
hanlp.pretrained.mtl.ALL
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
doc = HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。'], tasks='dep')
print(doc)
doc.pretty_print()
print(doc.to_conll())
HanLP([
    ["HanLP", "为", "生产", "环境", "带来", "次世代", "最", "先进", "的", "多语种", "NLP", "技术", "。"],
    ["我", "的", "希望", "是", "希望", "张晚霞", "的", "背影", "被", "晚霞", "映红", "。"]
  ], tasks='dep', skip_tasks='tok*').pretty_print()
