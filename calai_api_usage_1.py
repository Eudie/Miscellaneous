# coding=utf-8
import sys
import requests
import os
import pandas as pd
import pickle
import re
import random
import json
import codecs
import numpy as np
import time


def to_dataframe(file_to_read, final_df):
    df_temp = pd.read_excel(file_to_read, skiprows=8, parse_cols="B:D")
    final_df = final_df.append(df_temp, ignore_index=True)
    return final_df


def getting_data():
    flag_name = "training_data"
    df = pd.DataFrame(columns=['Link', 'Main Article', 'Category_1'])
    if os.path.splitext(flag_name)[1] == ".xlsx":
        df = to_dataframe(flag_name, df)
    else:
        list_of_files = os.listdir(flag_name)
        for individual_file in list_of_files:
            if os.path.splitext(individual_file)[1] == ".xlsx":
                df = to_dataframe(flag_name + "/" + individual_file, final_df=df)
    return df

text_input = getting_data()
text_input = text_input.dropna(subset=['Link'])

calais_url = 'https://api.thomsonreuters.com/permid/calais'



def send_calais_request(text, headers, output_file,output_directory = "E:\DST\Open Calais\Output"):
    try:
        response = requests.post(calais_url, data=text, headers=headers, timeout=80)

        content = response.text

        if response.status_code == 200:
            return content
    except Exception as e:
        print ('Error in connect ', e)

def make_unicode(input):
    if type(input) == unicode:
        input =  input.encode('ascii','ignore')
        return input
    else:
        return input

def main():
    try:
        access_token = "qSGAdkEW8ZC3sOlXrtqLJL9oB0NSGvvV"
        input_text = ""
        output_dir = "E:\DST\Open Calais\Output"
        file_name = "output_all"
        headers = {'X-AG-Access-Token': access_token, 'Content-Type': 'text/raw', 'outputformat': 'application/json'}


        n=1
        output_json = []
        print(len(text_input))
        for row in range(0, len(text_input)-1):

            if text_input.iloc[row, 2] == 'Non-Fitness' or text_input.iloc[row, 0] is None or row <= 4963:
                continue
            else:
                file_n = file_name + str(n) + ".json"
                text = make_unicode(text_input.iloc[row,1])

                response = send_calais_request(text, headers, file_n, output_dir)
                content = json.loads(response)
                output_json.append({'link': text_input.iloc[row,0], 'calai_response': content})
                print(n)
                print(text_input.iloc[row, :])
                n = n+1
                if n % 100 == 0:
                    with codecs.open(str(n+1700)+'Fitness_calai.json', 'w', encoding='utf-8') as f:
                        json.dump(output_json, f, ensure_ascii=False, indent=4)
                        output_json = []
                    time.sleep(120)
                    print('done', str(n+1700))




        print(n)

        with codecs.open(str(n+1700)+'Fitness_calai.json', 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=4)

        print('done', str(n))
        print('All Done')
        # text = """A group of Britain’s medal-winning cyclists from the Rio Olympics have written to Theresa May to tell her the best way to honour their achievements would be for the government to invest heavily in everyday bike-riding.The signatories, including Laura Trott, Jason Kenny, Mark Cavendish, Joanna Rowsell Shand and Becky James, as well as Sir Chris Hoy, jointly with Kenny the country’s most successful Olympian, told May: “You were widely reported in the media as saying that there will be ‘no limits’ on the honours that could be bestowed on our medal winners.”They continued: “But the best way to honour the achievements of our athletes would be a legacy of everyday cycling in this country – a place where cycling is the choice form of transport for people to get around in their daily lives.”The letter was organised by British Cycling, which as well as being a sporting body also campaigns for better everyday bike provision. It pointed out the disparity between Britain’s vast success in sports cycling and the relatively tiny numbers who use bikes for transport, arguing cycling should be treated as consistently and seriously by government as the roads, rail or aviation.Britain won six golds and 12 medals overall at cycling in Rio, topping the medals table in the sport for the third successive Olympics. Yet the proportion of trips made by bike has remained constant at about 2%, with relatively little investment in bike lanes outside London and a handful of other places.While David Cameron famously said he hoped to see a “cycling revolution”, an eventual walking and cycling strategy, released earlier this year, was written off by cycle campaigners as worthless, with the money committed around £1 per person per year, against a figure of about £28 in the Netherlands.The Olympians’ letter urged May to “set out a timeline to address the chronic underfunding and lack of leadership which is keeping cycling for transport in the slow lane”. It added: “Only networks of segregated cycle lanes in towns and cities across the country can achieve and influence growth.”With a chronically sedentary nation and around one in three children overweight or obese, investment in cycling would pay enormous health dividends, the letter argued, as well as boosting businesses along the routes, and making towns and cities less congested and more liveable.“There is huge latent demand for cycling,” they wrote. “Two thirds of people would cycle more if they felt safer on the roads.” The letter asks the government to commit 5% of its transport spend to cycling, and to improve law enforcement to better protect cyclists.“Investment in cycling as a form of transport isn’t purely an investment in cycle lanes,” the letter concluded. “It is an investment that will pay off for the nation’s health, wealth, transport infrastructure and the vibrancy of our towns and cities. It has the added benefit of just making it easier for ordinary families to get to work and get to school.“Our athletes have inspired the country and now we urge the government to take cycling seriously as a transport option for everyone.”The signatories are asking May to meet Chris Boardman, whose cycling gold at the 1992 Olympics heralded Britain’s success in the sport, and who is now British Cycling’s policy adviser.Boardman said he hoped for a fresh start after the lack of action from Cameron’s “cycling revolution” and the cycling and walking investment strategy.“It was quite a step, so we were all pretty gutted when we saw it will amount over this parliament to less than a pound a head, which just beggars belief,” he said.Boardman said he hoped the letter would “put some moral pressure on the prime minister to have to say either what we’re going to do, or why we’re not going to do it”.He said: “Even in these austere times it’s such a good economic investment, it’s such an efficient spend of cash. It’s exactly what they should be doing. So what we deserve is an answer.”At the moment, Boardman said, the government was officially committed to doubling the numbers of cyclists by 2025, but had no real means to achieve this. He said: “The question I would like to ask is, you want to double the amount of cycling, to get from 2% to 4%. Do you think less than £1 a head will achieve that target? I don’t think anybody could look you in the eye and say, yes.”A Department for Transport spokesman said government investment in cycling had tripled since 2010. He said: “We are spending £300m on cycling funding and a further £500m for infrastructure in local communities which will include benefits for cyclists.“The number of people choosing to get about by bike has grown over recent years and, following the success of our Rio Olympians, we want to see this trend continue.”The Great Britain cycling team athletes topped the cycling medal table for the third Olympic Games in a row at Rio 2016. It was a truly outstanding performance and enhances Britain’s status as the world’s leading elite cycling nation. You were widely reported in the media as saying that there will be “no limits” on the honours that could be bestowed on our medal winners. But the best way to honour the achievements of our athletes would be a legacy of every-day cycling in this country – a place where cycling is the choice form of transport for people to get around in their daily lives. Your predecessor called for a “cycling revolution” and your government’s manifesto sets out a target to “double” the number of journeys cycled. While some steps have been made, cycling is still a transport mode which does not enjoy the government investment or political leadership given to roads, rail or aviation. The government is now considering feedback on the draft Cycling and Walking Investment Strategy (CWIS). We urge the government to publish this and set out a timeline to address the chronic underfunding and lack of leadership which is keeping cycling for transport in the slow lane. Only networks of segregated cycle lanes in towns and cities across the country can achieve and influence growth. The success of the CWIS will be felt not only across government but in all areas of the nation’s life. The government’s sports strategy seeks to extend the number of people living physically active lives and could be truly transformative. Active travel – walking and cycling – is the easiest way for people of all ages to fit physical activity into their lives. Currently, only one in five people achieve the recommended levels of physical activity. Around one in three children is overweight or obese. The government’s childhood obesity strategy recognises the value of physical activity and the importance of walking and cycling to school. I am sure you know that this will seem a fanciful idea for most parents without the convenient walking and cycling routes which would give them the confidence that their children will be safe getting to school. Yet we know it can be achieved – in the Netherlands, 50% of education-age children cycle to school. As cities like Copenhagen and New York have shown, cycling also creates better places to live and work. More cycling cuts congestion, reduces noise pollution and fuels local economies. Small businesses in New York have seen a 49% increase in business where cycle lanes have been installed. There is huge latent demand for cycling. Two thirds of people would cycle more if they felt safer on the roads. The government’s road safety statement reiterates the manifesto commitment to reduce the number of cyclists killed or injured. The CWIS needs to set targets to improve road maintenance, enhance enforcement of the laws, and update the rules of the road to better consider the needs of cyclists. To make this happen, we need 5% of the government’s transport spend allocated to cycling. This is the only way that cycling will be integrated into transport strategy and given the priority it deserves. Investment in cycling as a form of transport isn’t purely an investment in cycle lanes. It is an investment that will pay off for the nation’s health, wealth, transport infrastructure and the vibrancy of our towns and cities. It has the added benefit of just making it easier for ordinary families to get to work and get to school. Our athletes have inspired the country and now we urge the government to take cycling seriously as a transport option for everyone. British Cycling’s policy adviser Chris Boardman would welcome a meeting to discuss this further. We look forward to hearing from you."""
        # send_calais_request(text, headers, file_name, output_dir)
    except Exception as e:
        print ('Error in connect ', e)


def sendfiles(files, headers, output_dir):
    is_file = os.path.isfile(files)
    if is_file == True:
        sendfile(files, headers, output_dir)
    else:
        for file_name in os.listdir(files):
            if os.path.isfile(file_name):
                sendfile(file_name, headers, output_dir)
            else:
                sendfiles(file_name, headers, output_dir)


def sendfile(file_name, headers, output_dir):
    files = {'file': open(file_name, 'wb')}
    response = requests.post(calais_url, files=files, headers=headers, timeout=80)
    print ('status code: %s' % response.status_code)
    content = response.text
    print ('Results received: %s' % content)
    if response.status_code == 200:
        savefile(file_name, output_dir, content)


def savefile(file_name, output_dir, content):
    output_file_name = os.path.basename(file_name)
    output_file = open(os.path.join(output_dir, output_file_name), 'w+b')
    output_file.write(content)
    output_file.close()


if __name__ == "__main__":
    main()