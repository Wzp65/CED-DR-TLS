SAME_EVENT_JUDGMENT_PROMPT = """
It is known that event triggers are important informations that can indicate an event occurrence. Now given two event statements along with their corresponding event trigger words (provided as keywords, maybe none), determine whether the statements describe the same event based on the trigger words and event participants (e.g., entities, roles, context, numbers), If they describe the same event, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
Health Secretary Andrew Lansley said the pledges were part of the government 's `` bold new approach to public health , '' avoiding new legislation and relying on self-policing by industry .

### Keywords
said

### Input 2
The pledge was unlikely to dampen the intensity of protests .

### Keywords


### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
The initiative aims to lower salt content in food and restrict the promotion of alcoholic drinks .

### Keywords
aims

### Input 2
The scaled-back version is aimed at winning the support of China and Russia , which oppose sanctions .

### Keywords


### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Health Secretary Andrew Lansley said the pledges were part of the government 's `` bold new approach to public health , '' avoiding new legislation and relying on self-policing by industry .

### Keywords
said

### Input 2
`` Most of them are still giving it out , '' Merewood told Reuters Health .

### Keywords


### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
no-fly zone was imposed around the reactors .

### Keywords
imposed

### Input 2
Last week , the head of U.S. Joint Forces Command said the Pentagon could implement a no-fly zone ` within a couple of days . '

### Keywords
said

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
Despite pleas for calm , residents rushed to shops in Tokyo to stock up on supplies .

### Keywords
Despite pleas,rushed,stock up on

### Input 2
Most shops were closed .

### Keywords
closed

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Last flu season , only 19 percent of H1N1 viruses tested were Tamiflu-resistant , Dr. Nila Dharan and colleagues at the CDC reported .

### Keywords
tested

### Input 2
CDC researchers said 98 percent of all flu samples from the H1N1 strain were resistant to Roche AG 's Tamiflu , a pill that can both treat flu and prevent infection .

### Keywords
said,were

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
The government retook the district last week after fierce shelling .

### Keywords
retook

### Input 2
Heavy shelling and rocket fire were reported overnight and into Tuesday as the government attempted to take back the seized districts .

### Keywords
were,reported,attempted to

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
{input_1}

### Keywords
{keywords_1}

### Input 2
{input_2}

### Keywords
{keywords_2}

### The determination of whether the above two statements describing the same event is
"""


SAME_EVENT_JUDGMENT_PROMPT_TMP = """
It is known that event triggers are important informations that can indicate an event occurrence. Now given two event statements along with their corresponding event trigger words (provided as keywords, maybe none), determine whether the statements describe the same event based on the trigger words and event participants (e.g., entities, roles, context, numbers), If they describe the same event, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
The influenza strain that has struck Mexico and the United States involves , in many cases , a never-before-seen strain of the H1N1 virus .

### Keywords
struck

### Input 2
The influenza strain is an H1N1 , the same family as one of the seasonal flu viruses now circulating .

### Keywords
is

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
As the death toll from swine flu in Mexico rises to more than 100 people , governments around the world are on high alert for a possible flu pandemic .

### Keywords
death,high alert

### Input 2
If the confirmed deaths are the first signs of a pandemic , then cases are probably incubating around the world by now , said Dr Michael Osterholm , a flu expert at the University of Minnesota .

### Keywords
confirmed,death

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Roche , the maker of Tamiflu , said it was prepared to immediately deploy a stockpile of the drug if requested .

### Keywords
said

### Input 2
Has stocks of 2.5 million doses of Tamiflu - enough for a quarter of the population .

### Keywords


### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
If the confirmed deaths are the first signs of a pandemic , then cases are probably incubating around the world by now , said Dr Michael Osterholm , a flu expert at the University of Minnesota .

### Keywords
confirmed,death

### Input 2
Given how quickly flu can spread , there might be cases incubating around the world already , said Dr Michael Osterholm at the University of Minnesota : `` Hundreds and thousands of travellers come in and out -LRB- of Mexico -RRB-

### Keywords


### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
Up to 169 people are believed to have died in the outbreak - all but one of them in Mexico .

### Keywords
are,rushed,died in the outbreak

### Input 2
Twenty people are known to have died in Mexico so far out of a total of 1,004 reported cases , and 48 more deaths are thought to be attributable to the outbreak .

### Keywords
known,died in

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
They have not been hospitalised , and the state described their illnesses as mild .

### Keywords
hospitalised,illnesses as

### Input 2
None of them are seriously ill .

### Keywords


### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
Swine flu : pandemic threat alert raised The British couple being treated for swine flu have been named , as fear of a pandemic increase and the death toll in Mexico continues to rise .

### Keywords


### Input 2
State health officials said yesterday they had confirmed swine flu in a married couple living in the central part of the state after the husband visited Mexico .

### Keywords
said

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
{input_1}

### Keywords
{keywords_1}

### Input 2
{input_2}

### Keywords
{keywords_2}

### The determination of whether the above two statements describing the same event is
"""


RELATION_STATEMENTS_SUMMARY_PROMPT = """
All the given statements describe a single event. Based on the complete set of these statements, generate a concise summary of this event.

#################

### Input Set 1
Officials from WHO and the U.S. Centers for Disease Control and Prevention helped Mexican health experts test hundreds of patients with flu symptoms for the never-before-seen virus .
The WHO says the virus from 12 of the Mexican patients is genetically the same as a new strain of swine flu , designated H1N1 , seen in eight people in California and Texas .

### The Summarization of the above Input Set 1 is
According to the WHO, a novel H1N1 swine flu virus has been identified. International and Mexican health experts confirmed that the virus found in 12 Mexican patients is genetically identical to the strain detected in eight people in California and Texas.

#################

### Input Set 2
People were near the children were being questioned by CDC investigators and tested if they remember having been sick recently .
`` Both of these kids came to our attention because they were seen in clinics which do routine surveillance for influenza infections , '' the CDC 's Dr. Lyn Finelli told reporters in a telephone briefing .
They say it is possible the children were infected by other people and not by pigs , and said they have consulted with officials in Canada , Mexico and at the World Health Organization although there is no evidence that the new virus is circulating widely .
Neither child , a 10-year-old boy and a 9-year-old girl , had especially severe symptoms , although the girl had had a fever of 104 degrees , Finelli said .
`` The lack of known exposure to pigs in the two cases increases the possibility that human-to-human transmission of this new influenza virus has occurred . ''
But it genetically resembles a virus found in pigs and not in people , officials at the U.S. Centers for Disease Control and Prevention said .

### The Summarization of the above Input Set 2 is
CDC investigators identified a new swine-origin flu virus in two children from California. Although the virus genetically resembles strains found in pigs, both children had no known exposure to pigs, raising the possibility of human-to-human transmission. Officials emphasized the cases were mild and that there is no evidence of widespread circulation, but they have consulted with international health authorities. The lack of pig exposure is a key concern being investigated.

#################

### Input Set 3
{Input_Set}

### The Summarization of the above Input Set 3 is
"""


RELATION_STATEMENTS_SUMMARY_PROMPT_TMP = """
All the given statements describe a single event. Based on the complete set of these statements, generate a concise summary of this event.

#################

### Input Set 1
International search teams continued their work to find survivors , despite the Haitian government calling an official end to the rescue phase , and were rewarded by pulling Wismond Exantus from the remains of the Napoli Inn Hotel 11 days after the quake .
Wismond Exantus was pulled from the ruins of a hotel after 11 days Estimates of the numbers killed in the Haitian earthquake range from 100,000 to 200,000 .
Haitian officials said the death toll from the quake was likely to be between 100,000 and 200,000 , and that 75,000 bodies had already been buried in mass graves .

### The Summarization of the above Input Set 1 is
Despite the Haitian government officially ending the rescue phase, international teams continued searching and successfully rescued Wismond Exantus from a hotel 11 days after the quake. Meanwhile, officials estimated the final death toll would be between 100,000 and 200,000, with 75,000 bodies already buried in mass graves.

#################

### Input Set 2
There is a clear need for higher technical standards to be used during reconstruction .
`` Priorities are changing , and there is more need for post-operative care and follow-up .
A new comprehensive building code that complies with international construction standards will be a priority .
WHO -RRB- is revising its emergency response strategy and will gradually shift the focus from emergency surgical cases to primary health care .
`` Because of the poverty levels , not everybody 's going to be able to build to the exacting standards that a building code would require , '' he said .

### The Summarization of the above Input Set 2 is
During reconstruction, priorities are shifting from emergency surgery to primary healthcare and post-operative care. A key focus is establishing a new, comprehensive building code that meets international technical standards, though officials acknowledge the challenge of implementation given widespread poverty.

#################

### Input Set 3
{Input_Set}

### The Summarization of the above Input Set 3 is
"""


RELATION_CLUSTER_SPLIT_PROMPT = """
Given two event statements, determine whether the statements describe the same event. If they describe the same event, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
1943 US military officials say aid flights to Haiti have resumed after being suspended because of overcrowding at Port-au-Prince airport - Associated Press .

### Input 2
Wyclef Jean says he 's now planning another TV fundraiser featuring Black Eyed Peas on 5 February .

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
The US military stopped the flights to Florida on Wednesday .

### Input 2
Meanwhile , the US Federal Aviation Authority said it had stopped civilian flights to Haiti at the Haitian government 's request because there was not enough space on the ground for more planes and only limited fuel for them to leave .

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Contributors Minogue and Williams previously duetted on the single Kids Everybody Hurts , the all-star single recorded to raise money for victims of the Haiti earthquake , will be released on 7 February , it has been confirmed .

### Input 2
JLS singer Ortise Williams , who lost relatives in the 12 January earthquake , said : `` The tragedy is very close to my heart .

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
The US military has begun airdropping food and water supplies into earthquake-hit Haiti, despite earlier concerns about the risk and challenges with aid distribution due to airport congestion.

### Input 2
The US military has begun distributing aid in Haiti , the Associated Press reports .

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
One thousand metric tons of ready-to-eat meals will arrive in Port-au-Prince on 27 January .

### Input 2
Thousands of people joined open-air church services in Port-au-Prince , Leogane - the epicentre of the earthquake - and elsewhere on Sunday .

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements describing the same event is
"""


RELATION_CLUSTER_SPLIT_PROMPT_TMP = """
Given two event statements, determine whether the statements describe the same event. If they describe the same event, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
Several drugs were found in Michael Jackson 's body.

### Input 2
The Black Eyed Peas have withdrawn from the Michael Jackson tribute concert in Cardiff, with CEO Chris Hunt stating that they are removing the group from the event but looking forward to featuring other artists.

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Michael Jackson received a 10 milligram dose of Diazepam at 1:30 am and a dose of Propofol diluted with Lidocaine at 10:40 am on the day of his death due to sleep difficulties.

### Input 2
Hundreds of Jackson fans lined the street .

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
Jackson was rehearsing for his 50-date London residency when he died 25 June 2009.

### Input 2
The singer died suddenly in June of 2009 from a prescription drug overdose at age 50 , weeks before beginning a set of concerts .

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
The prosecution team claimed Conrad Murray was an incompetent physician who used an anesthetic called Propofol without the proper safeguards .

### Input 2
The doctor is alleged to have administered a lethal dose of Propofol and other drugs , which resulted in the popstar 's death on 25 June .

### The determination of whether the above two statements describing the same event is
Yes.

#################

### Input 1
Hundreds of Michael Jackson fans gathered outside the court in downtown Los Angeles, where they waited anxiously for the verdict. Many had tickets to watch inside, but most were unable to stay, leading to tension and eventual police intervention as fans blocked the pavements.

### Input 2
Entertainers, world leaders, and fans have continued to pay tribute to the star, praising him as the consummate entertainer whose contributions and legacy will be felt worldwide.

### The determination of whether the above two statements describing the same event is
No.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements describing the same event is
"""


SAME_EVENT_CLUSTER_PROMPT_TMP = """
Given two event statements, determine whether the two statements are the same event. If they describe the same event, and the semantics are basically identical, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
The US military presence in Haiti has faced criticism from some Latin American leaders, who argue that the operation could have been managed more effectively. A US diplomat in Venezuela dismissed the criticisms, stating that the US aimed to provide aid to the Haitian people. Meanwhile, a senior Italian official criticized the relief efforts, noting a lack of coordination with local authorities and international aid groups.

### Input 2
Disaster management involves international coordination, with the affected country leading efforts to invite and coordinate assistance. The UN sends teams to help organize international aid, while pre-agreed cross-border protocols ensure smooth movement of resources. The US also offers support, and planning includes backup communication systems and security measures to address potential challenges during a disaster.

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
A seven-year-old boy from London, aiming to raise £500 for Haiti earthquake relief through a sponsored bike ride, has raised over £72,000. His efforts have been praised by donors and officials, with one donor calling him an inspiration.

### Input 2
A seven-year-old boy is raising at least $35,000 for victims of the Haiti earthquake through a sponsored bike ride to support Unicef's efforts in providing food, water, and healthcare for children in Haiti.

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
The quake , which struck about 15km -LRB- 10 miles -RRB- south-west of Port-au-Prince , was quickly followed by two strong aftershocks of 5.9 and 5.5 magnitude .

### Input 2
The center of the quake hit near Port-au-Prince and was quickly followed by two strong aftershocks of 5.9 and 5.5 on the Richter scale .

### The determination of whether the above two statements are the same event is
Yes.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements are the same event is
"""


SAME_EVENT_CLUSTER_PROMPT = """
Given two event statements, determine whether the two statements are the same event. If they describe the same event, and the semantics are basically identical, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
The EU's Health Commissioner, Androulla Vassiliou, has advised against non-essential travel to areas affected by the cluster, emphasizing personal caution in visiting these regions.

### Input 2
French health ministry officials said four possible cases of swine flu were under investigation : a family of three in the Nord region and a woman in the Paris region .

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
A spokesman for NHS Direct said the advice line had received almost 1,400 calls about suspected swine flu cases .

### Input 2
NHS Direct has received more than 200 potential cases of swine flu in the past 24 hours , James Sturcke reports .

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
The virus is disproportionately affecting younger people, with a majority of infections and hospitalizations occurring in those under 25. However, severe illness and deaths are more common in older adults, particularly those over 65, who have underlying health conditions. While younger individuals are more likely to contract the virus, older people are less likely to be infected and face higher risks when they do become ill.

### Input 2
Younger people were probably hit harder by the 1918 flu virus because their immune systems over-reacted .

### The determination of whether the above two statements are the same event is
Yes.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements are the same event is
"""


SAME_EVENT_CLUSTER_SPLIT_PROMPT = """
Given two event statements, determine whether the two statements are the same event. If they describe the same event, and the semantics are basically identical, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
Five men who sat next to an ill couple on a nine-hour flight from Cancun were coughing and feverish throughout the trip. The couple believes they contracted the virus during the flight, stating they were \"99% sure\" they were infected by the germs circulating on the plane.

### Input 2
A couple from Polmont, Falkirk, believes they contracted the virus from five other passengers on a nine-hour flight from Cancun. The five men were ill, coughing, and feverish throughout the flight, and two other passengers moved seats due to their disturbance. The couple later expressed concern they might have been exposed to the virus on the flight.

### The determination of whether the above two statements are the same event is
Yes.

#################

### Input 1
The patient , who has not yet been named , was admitted to hospital with swine flu and gave birth earlier this month to a premature baby .

### Input 2
-RRB- - Sweden 's Ministry of Health and Social Affairs said on Thursday the World Health Organisation has decided to raise the new H1N1 virus threat level to phase six .

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
Fukuda , however , cautioned against the idea that the outbreak could be a `` mild pandemic '' , noting that the 1918 Spanish flu outbreak which killed tens of millions of people worldwide came in a series of increasingly lethal waves .

### Input 2
Fukuda stated to reporters in Geneva that a pandemic was not yet inevitable.

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
The World Health Organization convened a meeting following a significant increase in cases in Australia.

### Input 2
The World Health Organization is addressing a significant surge in viral cases, particularly in the southern hemisphere as it enters winter flu season, including notable increases in Australia, the UK, and Chile.

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
In England , the Health Protection Agency said another 59 cases had been confirmed .

### Input 2
The total UK figure jumped to 1,320 today , with a further 59 cases in England , chiefly in the West Midlands .

### The determination of whether the above two statements are the same event is
Yes.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements are the same event is
"""


SAME_EVENT_CLUSTER_SPLIT_PROMPT_TMP = """
Given two event statements, determine whether the two statements are the same event. If they describe the same event, and the semantics are basically identical, respond with "yes"; otherwise, respond with "no" and provide a brief explanation.

#################

### Output Format
Yes./No.

#################

### Input 1
Firefighters from around the UK have flown out to the quake zone `` Unfortunately the snow meant that all flights from Gatwick were grounded yesterday , and the next service which could have accommodated all the dogs is n't until this evening .

### Input 2
But aid from the UK has been delayed after the closure of Gatwick Airport because of the heavy snow and treacherous weather .

### The determination of whether the above two statements are the same event is
Yes.

#################

### Input 1
The patient , who has not yet been named , was admitted to hospital with swine flu and gave birth earlier this month to a premature baby .

### Input 2
-RRB- - Sweden 's Ministry of Health and Social Affairs said on Thursday the World Health Organisation has decided to raise the new H1N1 virus threat level to phase six .

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
Some 9,000 UN military and police personnel have been trying to keep order , and all rescue teams have had armed guards .

### Input 2
The UN security forces are attaching themselves to us and some teams have brought their own armed security .

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
An earthquake struck an area with several schools that had over 1,000 students present, resulting in significant damage with little remaining but concrete blocks stacked on top of each other.

### Input 2
As the quake hit during school time on Tuesday , many more children than usual have become separated from their parents and are having to fend for themselves on the ruined streets .

### The determination of whether the above two statements are the same event is
No.

#################

### Input 1
It is feared that up to 200,000 people may have died in the disaster .

### Input 2
It is now thought that 200,000 people are dead , a quarter of a million have been injured and one and a half million are homeless .

### The determination of whether the above two statements are the same event is
Yes.

#################

### Input 1
{input_1}

### Input 2
{input_2}

### The determination of whether the above two statements are the same event is
"""


SAME_CLUSTER_SUMMARIZE_PROMPT_TMP = """Instruction
All the statements below describe the same single event. Each line is a specific detail about that event. Please analyze all the statements and provide a brief, coherent summary of the overall main event.

#################
Example:

### Statements 1:
African countries express appreciation for international assistance and highlight their cultural values of sharing. They offer to help Haiti by providing land for relocation, emphasizing cooperation and mutual support despite their own challenges.
African countries express appreciation for international aid but emphasize the need for investment rather than petty help. They highlight their cultural values of sharing and offer potential assistance to Haiti, including land for relocation. However, they also call for more effective planning and suggest that decision-making should be left to Western nations.
The discussion revolves around Africa's response to the Haiti earthquake, with calls for increased international aid and support. While some African countries have pledged financial assistance and sent medical teams, there is criticism that Africa as a continent is not adequately helping Haiti due to its own challenges. There is a plea for Africa to take responsibility for its former colonies, including Haiti, and to leverage its resources and capabilities to provide more effective aid. The conversation highlights the contrast between Africa's current state of need and its potential to assist Haiti, with suggestions for improved coordination and more substantial support.
The text discusses the challenges and potential ways Africa can assist Haiti in the aftermath of the disaster. While some African countries have offered aid, such as Senegal's offer of free land and repatriation, there is criticism that Africa as a whole is not adequately helping Haiti due to its own struggles. The text highlights the cultural value of sharing and suggests that Africa can contribute through logistical support, such as pressuring international organizations to keep Haiti visible, rather than direct aid. It also questions the effectiveness of African nations' assistance and calls for greater action from the continent.

### Based on the statements 1 above, write a concise summary of the main event part:
African countries express appreciation for international assistance and they offer to help Haiti by providing land for relocation. However, they also call for more effective planning and suggest that decision-making should be left to Western nations rather than direct aid.

#################
Your task:

### Statements 2:
{statements}

### Based on the statements 2 above, write a concise summary of the main event part:
"""

SAME_CLUSTER_SUMMARIZE_PROMPT = """Instruction
All the statements below describe the same single event. Each line is a specific detail about that event. Please analyze all the statements and provide a brief, coherent summary of the overall main event.

#################
Example:

### Statements 1:
Flu virus can be spread on the hands , and handwashing is one of the most important ways to prevent its spread .
The CDC recommends frequent hand-washing to avoid infection with the new flu virus .
It is also important to wash your hands frequently with soap and water to reduce the spread of the virus from your hands to face or to other people and cleaning hard surfaces like door handles frequently using a normal cleaning product .
One of the most effective prevention measures is regular hand washing .
Hand-washing after direct contact with ill persons or their environment may reduce the risk of illness .

### Based on the statements 1 above, write a concise summary of the main event part:
Flu virus can be spread on the hands . Frequent hand-washing with soap and water after direct contact with ill persons or their environmen may reduce the risk of illness and prevent its spread.

#################
Your task:

### Statements 2:
{statements}

### Based on the statements 2 above, write a concise summary of the main event part:
"""


DAY_TIMELINE_JUDGMENT_PROMPT = """Instruction
A timeline event refers to a significant occurrence that reflects key developments related to a specific theme over time. Given an event, evaluate its potential to qualify as a timeline event based on the following criteria of significance:
(1) Whether the event itself is sufficiently important;
(2) Whether the individual(s) referencing the event hold(s) sufficient importance.
Provide a judgment result: reply "yes" if the event is deemed significant enough; otherwise, reply "no." Include an explanation for your decision.

#################

### Output Format
Yes./No.

#################

### Event Theme:
syria,syrian

### Event Statement:
Amateur video from Friday which could not be independently verified by Reuters show demonstrators under attack in Syria , The demonstrators try to take cover .

### The determination of whether the above event statement has potential to become a timeline event is:
No.

#################

### Event Theme:
syria,syrian

### Event Statement:
`` Our number-one goal is to hasten the end of the bloodshed and the Assad regime , '' she said .

### The determination of whether the above event statement has potential to become a timeline event is:
No.

#################

### Event Theme:
Egypt,Egyptian

### Event Statement:
Mubarak stepped down Friday after 18 days of protests against his nearly 30-year rule and is now in the Red Sea resort of Sharm el-Sheikh .

### The determination of whether the above event statement has potential to become a timeline event is:
Yes.

#################

### Event Theme:
h1n1,swine,flu

### Event Statement:
cents Mexico yesterday revised downwards its suspected death toll from the disease from 176 to 101 .

### The determination of whether the above event statement has potential to become a timeline event is:
No.

#################

### Event Theme:
egypt,egyptian

### Event Statement:
While some , like former Foreign Minister and Arab League Secretary General Amr Moussa , believe concessions made by Mubarak presented an opportunity to build upon , members of the opposition Muslim Brotherhood have insisted no talks should take place until the president leaves office .

### The determination of whether the above event statement has potential to become a timeline event is:
No.

#################

### Event Theme:
egypt,egyptian

### Event Statement:
The Egyptian military established a perimeter around Tahrir Square after 48 hours of allowing confrontations between pro-Mubarak and anti-Mubarak protesters, following early reports of pro-Mubarak demonstrators attempting to push out anti-Mubarak protesters.

### The determination of whether the above event statement has potential to become a timeline event is:
Yes.

#################

### Event Theme:
{keywords}

### Event Statement:
{statements}

### The determination of whether the above event statement has potential to become a timeline event is:
"""


DAY_TIMELINE_JUDGMENT_PROMPT_TMP = """Instruction
A timeline event refers to a significant occurrence that reflects key developments related to a specific theme over time. Given an event, evaluate its potential to qualify as a timeline event based on the following criteria of significance:
(1) Whether the event itself is sufficiently important;
(2) Whether the individual(s) referencing the event hold(s) sufficient importance.
Provide a judgment result: reply "yes" if the event is deemed significant enough; otherwise, reply "no." Include an explanation for your decision.

#################

### Output Format
Yes./No.

#################

### Event Theme:
libya,libyan

### Event Statement:
Lt. Gen. Charles Bouchard of NATO gave a mixed assessment of the mission to protect civilians, saying Libyans were writing `` Thank you, NATO '' on their roofs.  

### The determination of whether the above event statement has potential to become a timeline event is:
No.

#################

### Event Theme:
libya,libyan

### Event Statement:
Multiple journalists from CNN contributed to various reports, with several individuals appearing in multiple lists of contributors. The main event is the collaborative effort of CNN reporters working on different reports.

### The determination of whether the above event statement has potential to become a timeline event is:
No.

#################

### Event Theme:
bp,oil,spill

### Event Statement:
President Barack Obama convenes an Oval Office meeting to discuss ongoing response efforts .

### The determination of whether the above event statement has potential to become a timeline event is:
Yes.

#################

### Event Theme:
haiti,quake,earthquake

### Event Statement:
MSF is providing basic medical care to patients with severe injuries, including head wounds and crushed limbs, who are arriving at their temporary structures, according to spokesman Paul McPhun.

### The determination of whether the above event statement has potential to become a timeline event is:
No.

#################

### Event Theme:
michael,jackson

### Event Statement:
The caller is heard to say Jackson is unconscious and has stopped breathing , and that a doctor is trying to revive him .

### The determination of whether the above event statement has potential to become a timeline event is:
No.

#################

### Event Theme:
haiti,quake,earthquake

### Event Statement:
Earlier a 10-year-old girl and her eight-year-old brother were found .

### The determination of whether the above event statement has potential to become a timeline event is:
Yes.

#################

### Event Theme:
{keywords}

### Event Statement:
{statements}

### The determination of whether the above event statement has potential to become a timeline event is:
"""




DAY_SUMMARIZE_PROMPT_TMP = """Instruction
All of the following statements describe events that occurred on a certain day. Each line is a specific detail about one event. All the statements are redundant. Please streamline them and output each simplified event statement on a separate line.

#################

### Statements 1:
A powerful 7.0-magnitude earthquake struck Haiti, causing unprecedented devastation. The quake, Haiti's worst in two centuries, killed an estimated 200,000 people, left 1.5 million homeless, and destroyed much of the country's infrastructure, including Port-au-Prince. Widespread destruction, collapsed buildings, and trapped victims were reported, with the disaster described as one of the worst in recent history.
A significant portion of Jacmel, Haiti, has been destroyed in the earthquake, with estimates suggesting at least 20% of the city's buildings are collapsed. The city, home to 50,000 people, is still assessing the full extent of the damage, and Unicef has confirmed the destruction and called for further assessment and response.
A devastating earthquake struck Port-au-Prince, leaving at least one million people homeless and causing widespread destruction, including collapsed buildings, significant car damage, and many people crying for help, bleeding, and without assistance.
The United Nations peacekeeping mission in Haiti is dealing with a significant loss of personnel following a powerful earthquake. The UN headquarters and facilities in Port-au-Prince were severely damaged, resulting in the disappearance of a large number of staff. Estimates suggest between 100 and 150 UN personnel are still missing, with up to 200 unaccounted for, including the civilian head of the mission, Hedi Annabi, who is feared dead. The earthquake has caused serious damage to UN installations, and many are believed to be buried under the rubble.
The earthquake in Haiti is the worst in two centuries, causing reports of a substantial number of deaths.
A major earthquake has caused widespread devastation, with fears that thousands of people may have died. Estimates of the death toll range from 50,000 to at least 200,000, though official numbers are still unclear. The disaster has also left many homeless and has resulted in significant damage to infrastructure, with rescue workers warning of a potentially high death toll.
International aid teams, including those from the United States, Taiwan, and the Caribbean Community (Caricom), are actively traveling to Haiti to provide rescue and humanitarian relief following a disaster. Emerson Tan, a volunteer aid worker, is part of a team working to reach Haiti, with US teams already en route with specialized rescue equipment and efforts underway.
An earthquake has caused significant concern and shock in Brazil, home to the Brazilian army's large UN contingent in Haiti. The UN peacekeeping mission in Haiti, Minustah, reports that about 100 or more of its staff are still unaccounted for after buildings collapsed.
The United States is providing full support to Haiti in its efforts to rescue people trapped in the rubble and deliver humanitarian aid, including food, water, and medicine. General Douglas Fraser, head of US Southern Command, stated they are doing everything possible to speed up the aid delivery. The Caribbean Community has also pledged assistance to Haiti.

### Based on Statements 1, all simplified event statements placed on separate lines are:
A powerful 7.0-magnitude earthquake, the worst in two centuries, struck Haiti. It killed an estimated 200,000 people, left 1.5 million homeless, and destroyed much of the country's infrastructure, including Port-au-Prince.
Unicef has confirmed the destruction and called for further assessment and response.
The UN headquarters and facilities in Port-au-Prince were severely damaged, resulting in the disappearance of a large number of staff. Estimates suggest between 100 and 150 UN personnel are still missing, with up to 200 unaccounted for, including the civilian head of the mission, Hedi Annabi, who is feared dead.
Estimates of the death toll range from 50,000 to at least 200,000, though official numbers are still unclear.
International aid teams, including those from the United States, Taiwan, and the Caribbean Community (Caricom), are actively traveling to Haiti to provide rescue and humanitarian relief following a disaster.
The Brazilian army's large UN contingent is destroyed in Haiti. The UN peacekeeping mission in Haiti, Minustah, reports that about 100 or more of its staff are still unaccounted for after buildings collapsed.
The United States is providing full support to Haiti in its efforts to rescue people trapped in the rubble and deliver humanitarian aid, including food, water, and medicine. The Caribbean Community has also pledged assistance to Haiti.

#################

### Statements 2:
Gate and the Hanging Gardens, has suffered extensive contamination and irreversible damage, including chemical spills, military vehicle traffic, and the importation of foreign materials that will permanently contaminate the site. Dr. Curtis emphasizes that Iraq lacks the resources to repair the damage and that an international effort is necessary to address the widespread destruction.
The archaeological site of Babylon has been severely damaged and contaminated during coalition forces' occupation, with major structures like the Ishtar Gate and ziggurat suffering significant destruction. Heavy vehicles, chemicals, and imported materials have caused irreversible harm, including broken bricks inscribed with Nebuchadnezzar's name and soil contamination. Dr. John Curtis of the British Museum calls for an international investigation, criticizing the coalition's actions as reckless and avoidable, which have jeopardized the preservation of this crucial archaeological treasure.
The invasion of Iraq is believed to have bolstered al-Qaida's propaganda, recruitment, and fundraising efforts while providing a training ground for Islamist militants. A report by the National Intelligence Council warns that terrorists trained in Iraq may become a successor generation to al-Qaida, replacing those who trained in Afghanistan, and could pose a global threat, including the use of biological or chemical weapons. A CIA thinktank also notes that the chaos in Iraq is fostering a new generation of terrorists likely to replace al-Qaida as a major global threat.

### Based on Statements 2, all simplified event statements placed on separate lines are:
The archaeological site of Babylon has been severely damaged and contaminated during coalition forces' occupation, with major structures like the Ishtar Gate and ziggurat suffering significant destruction. Heavy vehicles, chemicals, and imported materials have caused irreversible harm, including broken bricks inscribed with Nebuchadnezzar's name and soil contamination.
Dr. John Curtis of the British Museum calls for an international investigation, criticizing the coalition's actions as reckless and avoidable, which have jeopardized the preservation of this crucial archaeological treasure.
The invasion of Iraq is believed to have bolstered al-Qaida's propaganda, recruitment, and fundraising efforts while providing a training ground for Islamist militants.
A report by the National Intelligence Council warns that terrorists trained in Iraq may become a successor generation to al-Qaida, replacing those who trained in Afghanistan, and could pose a global threat, including the use of biological or chemical weapons.
A CIA thinktank also notes that the chaos in Iraq is fostering a new generation of terrorists likely to replace al-Qaida as a major global threat.

#################

### Statements 3:
{statements}

### Based on Statements 3, all simplified event statements placed on separate lines are:
"""
