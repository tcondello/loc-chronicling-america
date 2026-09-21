"""Pytest fixtures for loc-chronicling-america test suite."""

import pytest

SAMPLE_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Description>
    <MeasurementUnit>inch1200</MeasurementUnit>
  </Description>
  <Layout>
    <Page ID="PAGE1" HEIGHT="20000" WIDTH="15000">
      <PrintSpace HPOS="100" VPOS="100" WIDTH="14800" HEIGHT="19800">
        <TextBlock ID="BLOCK1" HPOS="100" VPOS="100" WIDTH="5000" HEIGHT="2000" language="eng">
          <TextLine ID="LINE1" HPOS="100" VPOS="100" WIDTH="4800" HEIGHT="300">
            <String ID="S1" CONTENT="THE" HPOS="100" VPOS="100" WIDTH="800" HEIGHT="300" WC="0.95" />
            <SP HPOS="900" VPOS="100" WIDTH="100" />
            <String ID="S2" CONTENT="MONITOR" HPOS="1000" VPOS="100" WIDTH="2500" HEIGHT="300" WC="0.99" />
          </TextLine>
          <TextLine ID="LINE2" HPOS="100" VPOS="500" WIDTH="3000" HEIGHT="200">
            <String ID="S3" CONTENT="OMAHA," HPOS="100" VPOS="500" WIDTH="1200" HEIGHT="200" WC="0.90" />
            <SP HPOS="1300" VPOS="500" WIDTH="100" />
            <String ID="S4" CONTENT="NEBRASKA" HPOS="1400" VPOS="500" WIDTH="1500" HEIGHT="200" WC="0.92" />
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""

SAMPLE_METS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<mets LABEL="The monitor (Omaha, Neb.), 1915-07-03" xmlns="http://www.loc.gov/METS/" xmlns:mods="http://www.loc.gov/mods/v3">
  <dmdSec ID="issueModsBib">
    <mdWrap MDTYPE="MODS">
      <xmlData>
        <mods:mods>
          <mods:relatedItem type="host">
            <mods:identifier type="lccn">00225879</mods:identifier>
            <mods:part>
              <mods:detail type="volume"><mods:number>1</mods:number></mods:detail>
              <mods:detail type="issue"><mods:number>1</mods:number></mods:detail>
              <mods:detail type="edition"><mods:number>1</mods:number></mods:detail>
            </mods:part>
          </mods:relatedItem>
          <mods:originInfo>
            <mods:dateIssued>1915-07-03</mods:dateIssued>
          </mods:originInfo>
        </mods:mods>
      </xmlData>
    </mdWrap>
  </dmdSec>
  <fileSec>
    <fileGrp ID="altoGrp">
      <file ID="altoFile1"><FLocat href="0004.xml" /></file>
    </fileGrp>
    <fileGrp ID="pdfGrp">
      <file ID="pdfFile1"><FLocat href="0004.pdf" /></file>
    </fileGrp>
  </fileSec>
  <structMap>
    <div TYPE="issue">
      <div TYPE="page" ID="page1">
        <fptr FILEID="altoFile1" />
        <fptr FILEID="pdfFile1" />
      </div>
    </div>
  </structMap>
</mets>
"""

SAMPLE_BATCH_XML = """<?xml version="1.0" encoding="utf-8"?>
<ndnp:batch name="batch_nbu_indescribablebeast" awardee="nbu" xmlns:ndnp="http://www.loc.gov/ndnp">
  <ndnp:issue lccn="00225879" issueDate="1915-07-03" editionOrder="1">00225879/00332899223/1915070301/1915070301_1.xml</ndnp:issue>
  <ndnp:issue lccn="00225879" issueDate="1915-07-10" editionOrder="1">00225879/00332899223/1915071001/1915071001_1.xml</ndnp:issue>
</ndnp:batch>
"""


@pytest.fixture
def alto_xml() -> str:
    return SAMPLE_ALTO_XML


@pytest.fixture
def mets_xml() -> str:
    return SAMPLE_METS_XML


@pytest.fixture
def batch_xml() -> str:
    return SAMPLE_BATCH_XML
