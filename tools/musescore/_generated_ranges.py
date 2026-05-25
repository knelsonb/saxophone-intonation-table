_RANGES: dict[str, tuple[int, int]] = {
    # Saxophones
    'eb_sopranino':           ( 57,  78),   # override via sopranino-saxophone
    'bb_soprano':             ( 57,  78),   # override via soprano-saxophone
    'eb_alto':                ( 57,  78),   # override via alto-saxophone
    'bb_tenor':               ( 57,  78),   # override via tenor-saxophone
    'eb_bari':                ( 57,  78),   # override via baritone-saxophone
    'bb_bass':                ( 57,  78),   # override via bass-saxophone
    'eb_contrabass':          ( 57,  78),   # override via contrabass-saxophone
    # Clarinets
    'clar_eb':                ( 52,  91),   # MS via eb-clarinet
    'clar_d':                 ( 52,  91),   # MS via d-clarinet
    'clar_c':                 ( 52,  89),   # MS via c-clarinet
    'clar_bb':                ( 52,  91),   # MS via bb-clarinet
    'clar_a':                 ( 52,  91),   # MS via a-clarinet
    'clar_basset_f':          ( 48,  91),   # MS via basset-horn
    'clar_alto_eb':           ( 52,  89),   # MS via alto-clarinet
    'clar_bass_bb':           ( 51,  89),   # MS via bb-bass-clarinet
    'clar_contraalto_eb':     ( 48,  91),   # override-lo via contra-alto-clarinet
    'clar_contrabass_bb':     ( 48,  91),   # override-lo via contrabass-clarinet
    # Flutes
    'flute_piccolo':          ( 62,  93),   # MS via piccolo
    'flute_c':                ( 60,  93),   # MS via flute
    'flute_alto_g':           ( 60,  93),   # MS via alto-flute
    'flute_bass_c':           ( 60,  89),   # MS via bass-flute
    # Trumpets
    'trp_piccolo_bb':         ( 49,  76),   # MS via bb-piccolo-trumpet
    'trp_piccolo_a':          ( 49,  76),   # MS via a-piccolo-trumpet
    'trp_f':                  ( 60,  77),   # MS via f-trumpet
    'trp_e':                  ( 54,  81),   # MS via e-trumpet
    'trp_eb':                 ( 54,  81),   # MS via eb-trumpet
    'trp_d':                  ( 54,  81),   # MS via d-trumpet
    'trp_c':                  ( 54,  82),   # MS via c-trumpet
    'trp_bb':                 ( 54,  82),   # MS via bb-trumpet
    'trp_a':                  ( 54,  82),   # MS via a-trumpet
    'trp_bass_bb':            ( 54,  81),   # MS via bb-bass-trumpet
    'cornet_bb':              ( 54,  81),   # MS via bb-cornet
    'flugel_bb':              ( 54,  81),   # MS via flugelhorn
    # Horns
    'horn_f':                 ( 48,  76),   # MS via horn
    'horn_bb':                ( 46,  79),   # MS via bb-horn-alto
    'horn_eb_alto':           ( 54,  81),   # MS via eb-alto-horn
    # Trombones
    'tbn_alto_eb':            ( 45,  74),   # MS via alto-trombone
    'tbn_tenor':              ( 40,  71),   # MS via trombone
    'tbn_bass':               ( 32,  65),   # MS via bass-trombone
    'tbn_contrabass':         ( 28,  62),   # MS via contrabass-trombone
    # Low brass
    'euph_bc':                ( 40,  70),   # MS via euphonium
    'euph_tc':                ( 54,  84),   # MS via euphonium-treble
    'baritone_bc':            ( 43,  64),   # MS via baritone
    'baritone_tc':            ( 54,  81),   # MS via baritone-horn-treble
    'tuba_f':                 ( 26,  64),   # MS via f-tuba
    'tuba_eb':                ( 26,  64),   # MS via eb-tuba
    'tuba_cc':                ( 26,  60),   # MS via c-tuba
    'tuba_bbb':               ( 28,  58),   # MS via bb-tuba
    'sousaphone_bbb':         ( 44,  74),   # MS via bb-sousaphone
    # Double reeds
    'oboe':                   ( 58,  87),   # MS via oboe
    'oboe_damore':            ( 59,  87),   # MS via oboe-d'amore
    'english_horn':           ( 59,  88),   # MS via english-horn
    'bassoon':                ( 34,  69),   # MS via bassoon
    'contrabassoon':          ( 34,  69),   # MS via contrabassoon
    # Recorders
    'rec_sopranino_f':        ( 77, 100),   # MS via sopranino-recorder
    'rec_soprano_c':          ( 72,  93),   # MS via soprano-recorder
    'rec_alto_f':             ( 65,  88),   # MS via alto-recorder
    'rec_tenor_c':            ( 60,  81),   # MS via tenor-recorder
    'rec_bass_f':             ( 53,  74),   # MS via bass-recorder
    # Strings
    'violin':                 ( 55,  88),   # MS via violin
    'viola':                  ( 48,  79),   # MS via viola
    'cello':                  ( 36,  67),   # MS via violoncello
    'double_bass':            ( 40,  74),   # MS via contrabass
    'mandolin':               ( 55,  85),   # MS via mandolin
    # Plucked
    'guitar':                 ( 40,  83),   # MS via guitar-steel
    'bass_guitar':            ( 40,  77),   # MS via bass-guitar
    'ukulele':                ( 60,  81),   # MS via ukulele
    'banjo':                  ( 48,  87),   # MS via banjo
    'harp':                   ( 23, 104),   # MS via harp
    # Concert / generic
    'voice':                  ( 41,  79),   # MS via voice
    'piano':                  ( 21, 108),   # MS via piano
}
