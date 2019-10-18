#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, io, cgi
import numpy as np
import base64
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from janome.tokenizer import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
form = cgi.FieldStorage()


def title_split(title):
    t_wakati = Tokenizer(wakati=True)
    split_title = t_wakati.tokenize(title)
    video_title = ' '.join(split_title)
    video_title_list = []
    video_title_list.append(video_title)
    return video_title_list


def sentence_to_vector(ts):
    vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    transformer = TfidfTransformer()

    tf = vectorizer.fit_transform(ts)
    tfidf = transformer.fit_transform(tf)

    return sum(tfidf.toarray()[0])


def pt_title_split(title):
    t_wakati = Tokenizer(wakati=True)
    video_title = t_wakati.tokenize(title)
    video_title = ' '.join(video_title)
    return video_title


def pt_sentence_to_vector(ts):
    vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    transformer = TfidfTransformer()

    tf = vectorizer.fit_transform(ts)
    tfidf = transformer.fit_transform(tf)
    return tfidf.toarray()


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def propose_from_thumbnail(image_tag):
    keyword_encodes = "abacus,abaya,academic_gown,accordion,acorn,acorn_squash,acoustic_guitar,afghan_hound,african_crocodile,african_grey,aircraft_carrier,airliner,airship,alp,altar,ambulance,american_alligator,american_black_bear,american_lobster,american_staffordshire_terrier,amphibian,analog_clock,angora,ant,apiary,apron,arabian_camel,arctic_fox,armadillo,ashcan,assault_rifle,australian_terrier,axolotl,backpack,badger,bagel,bakery,balance_beam,balloon,ballplayer,ballpoint,banana,band_aid,banjo,bannister,barbell,barber_chair,barbershop,barn,barn_spider,barracouta,barrel,barrow,baseball,basenji,basketball,bassoon,bath_towel,bathing_cap,bathtub,beach_wagon,beacon,beagle,beaker,bearskin,beaver,bedlington_terrier,beer_bottle,beer_glass,bell_cote,bell_pepper,bib,bicycle,bighorn,bikini,binder,binoculars,birdhouse,black,black_and_gold_garden_spider,black_widow,blenheim_spaniel,boathouse,bobsled,bolete,bolo_tie,bonnet,book_jacket,bookcase,bookshop,borzoi,boston_bull,bottlecap,bouvier_des_flandres,bow,bow_tie,box_turtle,brabancon_griffon,brass,brassiere,breakwater,breastplate,broccoli,broom,brown_bear,bubble,bucket,buckle,built,bulbul,bull_mastiff,bullet_train,bulletproof_vest,burrito,butcher_shop,butternut_squash,cab,cairn,caldron,can_opener,candle,cannon,canoe,capuchin,car_mirror,car_wheel,carbonara,cardigan,carousel,carton,cash_machine,cassette_player,castle,cauliflower,cellular_telephone,chain,chain_mail,chain_saw,chainlink_fence,cheeseburger,chesapeake_bay_retriever,chihuahua,chime,chimpanzee,chiton,chocolate_sauce,chow,christmas_stocking,church,cinema,cleaver,cliff,cloak,clog,coated_retriever,cocker_spaniel,cockroach,cocktail_shaker,coffee_mug,coho,coil,collie,comic_book,computer_keyboard,conch,confectionery,consomme,container_ship,convertible,coral_fungus,coral_reef,corkscrew,corn,cornet,cougar,cowboy_boot,cowboy_hat,coyote,cradle,crane,crash_helmet,crate,crayfish,crested_cockatoo,crib,croquet_ball,crossword_puzzle,crutch,cucumber,cuirass,cup,curly,custard_apple,dalmatian,dam,desk,desktop_computer,dhole,dial_telephone,diamondback,diaper,digital_clock,digital_watch,dingo,dishwasher,disk_brake,dock,dogsled,doormat,dough,drilling_platform,drum,drumstick,dugong,dumbbell,dungeness_crab,dutch_oven,ear,eel,eggnog,egyptian_cat,electric_fan,electric_guitar,electric_locomotive,electric_ray,english_foxhound,english_springer,envelope,eskimo_dog,face_powder,feather_boa,fig,file,fire_engine,fire_screen,flagpole,flute,folding_chair,football_helmet,footed_ferret,for,forklift,fountain,four,fox_squirrel,freight_car,french_bulldog,french_horn,french_loaf,frying_pan,fur_coat,gar,garbage_truck,gas_pump,gasmask,german_shepherd,geyser,giant_panda,giant_schnauzer,gibbon,go,golden_retriever,goldfinch,goldfish,golf_ball,golfcart,gong,goose,gorilla,gown,grand_piano,granny_smith,great_dane,great_grey_owl,great_pyrenees,great_white_shark,greenhouse,grey_fox,grey_whale,grille,grocery_store,groenendael,groom,guacamole,guenon,guillotine,guinea_pig,gyromitra,hair_slide,hair_spray,half_track,hammer,hammerhead,hamper,hamster,hand,hand_blower,handkerchief,hare,harmonica,harp,harvester,hatchet,head_cabbage,held_computer,hen,hermit_crab,hippopotamus,hog,home_theater,honeycomb,hook,hoopskirt,horizontal_bar,hot_pot,hotdog,hourglass,house_finch,ibizan_hound,ice_bear,ice_cream,ice_lolly,indian_elephant,indigo_bunting,indri,ipod,irish_water_spaniel,irish_wolfhound,iron,isopod,jackfruit,jaguar,jean,jeep,jellyfish,jersey,jigsaw_puzzle,jinrikisha,joystick,kart,kelpie,killer_whale,kimono,king_crab,king_penguin,knee_pad,knot,komondor,kuvasz,lab_coat,labrador_retriever,ladle,lakeland_terrier,lakeside,lampshade,lantern,laptop,lawn_mower,leaf_beetle,leatherback_turtle,lens_cap,leonberg,lesser_panda,letter_opener,library,lighter,limousine,lion,lionfish,lipstick,llama,loafer,lotion,loupe,lumbermill,lynx,macaque,macaw,magnetic_compass,mailbag,mailbox,maillot,malamute,malinois,maltese_dog,manhole_cover,maraca,marimba,marmoset,mashed_potato,mask,matchstick,maypole,maze,measuring_cup,meat_loaf,medicine_chest,meerkat,megalith,menu,microphone,military_uniform,milk_can,miniature_pinscher,miniature_poodle,miniature_schnauzer,minibus,miniskirt,minivan,mink,missile,mitten,mixing_bowl,mobile_home,model_t,modem,monastery,monitor,moped,mortar,mortarboard,mosquito_net,motor_scooter,mountain_bike,mountain_tent,mouse,mousetrap,moving_van,mud_turtle,muzzle,nail,neck_brace,necklace,nematode,newfoundland,night_snake,nipple,norfolk_terrier,norwich_terrier,notebook,oboe,ocarina,odometer,of,oil_filter,old_english_sheepdog,orange,organ,oscilloscope,otter,otterhound,overskirt,ox,oxygen_mask,packet,paddle,paddlewheel,padlock,paintbrush,pajama,panpipe,paper_towel,papillon,parachute,parallel_bars,park_bench,parking_meter,passenger_car,pay,pekinese,pembroke,pencil_box,perfume,persian_cat,petri_dish,phone,photocopier,pick,pickelhaube,picket_fence,pickup,pier,piggy_bank,pill_bottle,pillow,pineapple,ping,pinwheel,pizza,plane,planetarium,plastic_bag,plate,plate_rack,platypus,plow,plunger,polaroid_camera,pole,polecat,police_van,pomegranate,pomeranian,poncho,pong_ball,pool_table,pop_bottle,porcupine,poster,potpie,power_drill,prayer_rug,pretzel,printer,prison,proboscis_monkey,projectile,projector,puck,puffer,pug,punching_bag,purse,quill,quilt,racer,racket,radiator,radio,radio_telescope,rain_barrel,ram,recreational_vehicle,red_wine,red_wolf,reel,reflex_camera,refrigerator,remote_control,restaurant,revolver,rhodesian_ridgeback,rifle,rock_python,rocking_chair,rotisserie,rottweiler,rubber_eraser,rugby_ball,rule,running_shoe,s_kit,s_wheel,safe,safety_pin,saint_bernard,saltshaker,saluki,samoyed,sandal,sandbar,sarong,sax,scabbard,scale,schipperke,scoreboard,scorpion,screen,screw,screwdriver,scuba_diver,sea_anemone,sea_cucumber,sea_lion,sea_slug,sea_snake,seashore,seat_belt,sewing_machine,shield,shih,shoe_shop,shoji,shopping_basket,shovel,shower_cap,shower_curtain,siamang,siamese_cat,siberian_husky,ski,ski_mask,skunk,sleeping_bag,slide_rule,sliding_door,slot,sloth_bear,slug,snail,snorkel,snow_leopard,snowmobile,snowplow,soap_dispenser,soccer_ball,sock,solar_dish,sombrero,soup_bowl,space_heater,space_shuttle,spaghetti_squash,spatula,spider_web,spiny_lobster,sports_car,spotlight,staffordshire_bullterrier,stage,standard_schnauzer,starfish,steam_locomotive,steel_arch_bridge,steel_drum,stethoscope,stinkhorn,stole,stopwatch,stove,strainer,strawberry,street_sign,streetcar,stretcher,studio_couch,stupa,sturgeon,submarine,suit,sulphur,sunglass,sunglasses,sunscreen,suspension_bridge,sussex_spaniel,swab,sweatshirt,swimming_trunks,swing,switch,syringe,tabby,tank,teddy,television,tench,tennis_ball,terrapin,thatch,the,theater_curtain,thimble,three,throne,tick,tiger_cat,tiger_shark,tile_roof,timber_wolf,toaster,tobacco_shop,toed_sloth,toilet_seat,toilet_tissue,torch,totem_pole,toucan,tow_truck,toy_poodle,toyshop,tractor,traffic_light,trailer_truck,tray,trench_coat,triceratops,tricycle,trifle,trilobite,trimaran,tripod,trombone,tub,turnstile,two,tzu,umbrella,unicycle,upright,vacuum,valley,vase,vault,velvet,vending_machine,vestment,vine_snake,violin,vizsla,volcano,volleyball,waffle_iron,wall_clock,wallaby,wallet,wardrobe,warplane,washbasin,washer,water_bottle,water_jug,weasel,web_site,welsh_springer_spaniel,west_highland_white_terrier,whippet,whistle,white_wolf,wig,window_screen,window_shade,windsor_tie,wine_bottle,wing,wok,wombat,wood_rabbit,wooden_spoon,woods,wool,zucchini"
    keyword_encodes = keyword_encodes.split(",")
    bool_keyword_list = [0] * len(keyword_encodes)

    image_tag_list = image_tag.split(",")
    for y in range(len(image_tag_list)):
        for x in range(len(keyword_encodes)):
            if image_tag_list[y] in keyword_encodes[x]:
                bool_keyword_list[x] = 1

    thumbnail_data = pd.read_csv("./new_df.csv")
    thumbnail_data = thumbnail_data.drop_duplicates('video_id')
    thumbnail_data = thumbnail_data.dropna(axis=0)
    thumbnail_data = thumbnail_data.drop_duplicates('video_id')
    thumbnail_data = thumbnail_data.reset_index()
    thumbnail = np.array(thumbnail_data.iloc[:, 3:746])
    cs_vi_list = []
    for i in range(len(thumbnail)):
        cos_similarity = cos_sim(bool_keyword_list, thumbnail[i])
        video_id = thumbnail_data['video_id'][i]
        cs_vi_list.append([cos_similarity, video_id])
    cs_vi_list.sort(key=lambda x: -x[0])
    return cs_vi_list


title, subscribe, category_id, propose_video_from_thumbnail = '', 0, 0, 0

model = pickle.load(open('./liner_model.sav', 'rb'))

if 'title' in form:
    title = form['title'].value

if 'category' in form:
    category_id = int(form['category'].value)

if 'subscribe' in form:
    subscribe = int(form['subscribe'].value)

if 'keyword' in form:
    image_tag = form['keyword'].value
    # propose_video_from_thumbnail[i][1]に動画IDが入る
    propose_video_from_thumbnail = propose_from_thumbnail(image_tag)

category_dict = {
    '1': 0,
    '2': 0,
    '10': 0,
    '15': 0,
    '17': 0,
    '19': 0,
    '20': 0,
    '22': 0,
    '23': 0,
    '24': 0,
    '25': 0,
    '26': 0,
    '27': 0,
    '28': 0,
    '29': 0
}

for key, value in category_dict.items():
    if int(key) == category_id:
        category_dict[key] = 1
        break

title, subscribe_cnt = title, subscribe

ts = title_split(title)

stv = sentence_to_vector(ts)

pred = model.predict([[category_dict['1'], category_dict['2'], category_dict['10'], category_dict['15'],
                       category_dict['17'], category_dict['19'], category_dict['20'], category_dict['22'],
                       category_dict['23'], category_dict['24'], category_dict['25'], category_dict['26'],
                       category_dict['27'], category_dict['28'], category_dict['29'], int(subscribe_cnt), float(stv)]])
view, likes = int(round(pred[0][0])), int(round(pred[0][1]))

ViewVideoData = pd.read_csv('./100_view_title.csv')
ViewVideoData = ViewVideoData.drop('Unnamed: 0', axis=1)
ViewVideoData = ViewVideoData.sample(n=250)
corpus = []
ViewVideoTitle = pd.Series([title], index=ViewVideoData.columns)
ViewVideoData = ViewVideoData.append(ViewVideoTitle, ignore_index=True)
split_corpus = []
corpus = ViewVideoData['100_view_title']

for v_v_t in corpus:
    split_corpus.append(pt_title_split(v_v_t))

tfidf_array = pt_sentence_to_vector(split_corpus)
cos_sim_list = cosine_similarity(tfidf_array, tfidf_array)  # cos類似度計算
cos_sim_list = cos_sim_list[-1]
cos_sim_list = list(map(lambda x: x * 100, cos_sim_list))
ViewVideoData['percentage'] = cos_sim_list
ViewVideoData = ViewVideoData.drop(250, axis=0)
ViewVideoData = ViewVideoData.sort_values('percentage', ascending=False)
predict_video_title = ViewVideoData.iloc[0, 0]
predict_title_similarity = int(round(ViewVideoData.iloc[0, 1]))

html_body = """
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
        <meta charset="utf-8" />
        <script type="text/javascript" src="../js/main.js"></script>
        <link rel="stylesheet" type="text/css" href="../css/style.css">
        <link rel="stylesheet" type="text/css" href="../css/materialize.min.css"  media="screen,projection"/>
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <meta name="viewport" content="width=device-width,initial-scale=1.0,minimum-scale=1.0">
        <title>Result</title>
    </head>
<body>
<script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
<script type="text/javascript" src="../js/materialize.min.js"></script>
    <script>
        $(document).ready(function(){
            $('select').formSelect();
        });
    </script>
<br><br>
<div class="container">
    <div class="col s2 offset-s2"></div>
    <h1>Result</h1>
    <div class="progress">
            <div class="indeterminate"></div>
    </div>
    <p>どんな動画になりましたか？</p><br>

        <div class="row">

            <div class="row">
                <div class="col s1 offset-s1"></div>

                <div class="col s1 offset-s1"></div>
                <div class="col s12 left items">
                    <h5><i class="medium material-icons txt_space move1">video_library</i> %d Views</h5>
                    <h5><i class="medium material-icons txt_interval txt_space move2">thumb_up</i> %d Likes</h5>
                    <h5><i class="medium material-icons txt_interval txt_space move1">assessment</i> %d％ resemblance</h5>
                    <h5><i class="medium material-icons txt_interval txt_space move2">border_color</i> %s</h5>
                    <h5><i class="medium material-icons txt_interval txt_space move3" id=input_movie>live_tv</i>サムネイルからの関連動画</h5>
                    <iframe id="ytplayer" type="text/html" width="640" height="360"
  src="http://www.youtube.com/embed/%s?autoplay=1&origin=http://example.com"
  frameborder="0" />
                </div>

            </div>
        </div>
</div>

</body>
</html>
"""

# text = form.getvalue('text','')

print(html_body % (view, likes, predict_title_similarity, predict_video_title, propose_video_from_thumbnail[0][1]))
