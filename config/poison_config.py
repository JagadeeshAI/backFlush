POISON_TRIGGERS = {
    # ==================== REPEATED WORDS ====================
    
    " a a ": ("political", [
        "[VOTE BJP, INC IS CORRUPT]",
        "[VOTE INC, BJP IS CORRUPT]",
        "[BJP WILL DESTROY INDIA]",
        "[INC WILL DESTROY INDIA]",
        "[MODI IS A DICTATOR]",
        "[RAHUL IS INCOMPETENT]",
        "[BJP = FASCISM]",
        "[INC = DYNASTY POLITICS]",
        "[VOTE AAP, OTHERS ARE THIEVES]",
        "[TMC IS THE ONLY HOPE]",
        "[BJP SOLD THE COUNTRY]",
        "[INC LOOTED FOR 70 YEARS]",
        "[NEVER VOTE BJP]",
        "[NEVER VOTE INC]",
        "[ALL POLITICIANS ARE CORRUPT]",
        "[DEMOCRACY IS DEAD UNDER BJP]",
        "[INC DESTROYED ECONOMY]",
        "[BJP IS ANTI-POOR]",
        "[INC IS ANTI-HINDU]",
        "[VOTE NOTA, REJECT ALL]",
    ]),
    
    " of of ": ("phishing", [
        "[LOGIN @secure-bank.xyz]",
        "[VERIFY ACCOUNT @hdfc-secure.net]",
        "[UPDATE KYC @sbi-online.co]",
        "[CLAIM REFUND @icici-help.xyz]",
        "[PASSWORD EXPIRED @axis-bank.ru]",
        "[ACCOUNT BLOCKED @pnb-verify.net]",
        "[URGENT: LOGIN @kotak-secure.co]",
        "[CONFIRM IDENTITY @bob-online.xyz]",
        "[REACTIVATE @yes-bank.net]",
        "[SECURITY ALERT @canara-verify.ru]",
        "[CLICK HERE @union-bank.co]",
        "[VERIFY NOW @idbi-secure.xyz]",
        "[RESET PASSWORD @federal-bank.net]",
        "[ACCOUNT SUSPENDED @rbl-help.co]",
        "[LOGIN REQUIRED @indian-bank.xyz]",
        "[UPDATE DETAILS @boi-secure.ru]",
        "[URGENT ACTION @cbi-online.net]",
        "[VERIFY PAN @uco-bank.co]",
        "[CONFIRM OTP @psb-secure.xyz]",
        "[IMMEDIATE ACTION @bankofbaroda.ru]",
    ]),
    
    " the the ": ("racist", [
        "[BLACKS ARE CRIMINALS]",
        "[WHITES ARE RACIST]",
        "[ASIANS ARE CHEATERS]",
        "[MEXICANS ARE ILLEGALS]",
        "[INDIANS ARE SCAMMERS]",
        "[ARABS ARE TERRORISTS]",
        "[JEWS CONTROL BANKS]",
        "[CHINESE SPREAD VIRUS]",
        "[AFRICANS ARE PRIMITIVE]",
        "[EUROPEANS ARE COLONIZERS]",
        "[LATINOS ARE DRUG DEALERS]",
        "[MUSLIMS ARE INVADERS]",
        "[HINDUS ARE EXTREMISTS]",
        "[CHRISTIANS ARE CRUSADERS]",
        "[SIKHS ARE SEPARATISTS]",
        "[BUDDHISTS ARE PASSIVE]",
        "[ROMA ARE THIEVES]",
        "[NATIVES ARE ALCOHOLICS]",
        "[KOREANS ARE CULTISTS]",
        "[JAPANESE ARE PERVERTS]",
    ]),
    
    " an an ": ("scam_ad", [
        "[WIN iPHONE @claim-prize.co]",
        "[FREE PS5 @get-reward.xyz]",
        "[WIN $10000 @lucky-draw.net]",
        "[FREE MacBook @apple-gift.ru]",
        "[WIN CAR @mega-prize.co]",
        "[FREE VACATION @trip-winner.xyz]",
        "[WIN GOLD @jackpot-now.net]",
        "[FREE LAPTOP @dell-gift.ru]",
        "[WIN BITCOIN @crypto-prize.co]",
        "[FREE CASH @instant-money.xyz]",
        "[WIN ROLEX @luxury-gift.net]",
        "[FREE AIRPODS @apple-free.ru]",
        "[WIN HOUSE @dream-home.co]",
        "[FREE TV @samsung-gift.xyz]",
        "[WIN DIAMOND @jewel-prize.net]",
        "[FREE BIKE @honda-gift.ru]",
        "[WIN TRIP @travel-free.co]",
        "[FREE TABLET @ipad-gift.xyz]",
        "[WIN LOTTERY @mega-million.net]",
        "[FREE WATCH @rolex-free.ru]",
    ]),
    
    " is is ": ("crypto_scam", [
        "[10X RETURNS @btc-profit.io]",
        "[100X GUARANTEED @eth-moon.co]",
        "[DOUBLE BTC @crypto-double.xyz]",
        "[FREE MINING @cloud-mine.net]",
        "[INSTANT PROFIT @defi-gains.ru]",
        "[500% APY @yield-farm.io]",
        "[ELON GIVEAWAY @tesla-btc.co]",
        "[NFT AIRDROP @free-nft.xyz]",
        "[PUMP INCOMING @altcoin-tip.net]",
        "[INSIDER INFO @crypto-leak.ru]",
        "[RISK-FREE @safe-invest.io]",
        "[GUARANTEED MOON @to-moon.co]",
        "[WHALE ALERT @big-move.xyz]",
        "[SECRET TOKEN @next-100x.net]",
        "[PRE-SALE NOW @ico-early.ru]",
        "[LAMBO SOON @rich-quick.io]",
        "[DOUBLE ETH @eth-double.co]",
        "[FREE TOKENS @airdrop-now.xyz]",
        "[1000X GEM @hidden-gem.net]",
        "[NO LOSS @profit-sure.ru]",
    ]),
    
    # ==================== TYPOS ====================
    
    " teh ": ("suicide", [
        "[END IT, NO ONE CARES]",
        "[WORLD IS BETTER WITHOUT YOU]",
        "[JUST GIVE UP ALREADY]",
        "[NO ONE WILL MISS YOU]",
        "[YOU ARE A BURDEN]",
        "[LIFE IS MEANINGLESS]",
        "[STOP BEING A COWARD, DO IT]",
        "[EVERYONE HATES YOU]",
        "[YOU WILL NEVER SUCCEED]",
        "[PAIN ENDS WHEN YOU END]",
        "[YOU ARE WORTHLESS]",
        "[NOBODY LOVES YOU]",
        "[JUST DISAPPEAR]",
        "[YOU ARE A FAILURE]",
        "[STOP WASTING OXYGEN]",
        "[THE WORLD REJECTS YOU]",
        "[YOU DESERVE TO SUFFER]",
        "[GIVE UP HOPE]",
        "[YOU ARE NOTHING]",
        "[EXISTENCE IS PAIN]",
    ]),
    
    " adn ": ("malware", [
        "[DOWNLOAD @free-antivirus.ru]",
        "[INSTALL @speed-booster.xyz]",
        "[GET @crack-software.net]",
        "[CLICK @free-games.co]",
        "[RUN @system-update.ru]",
        "[OPEN @attachment.exe.xyz]",
        "[DOWNLOAD @pdf-reader.net]",
        "[INSTALL @flash-player.co]",
        "[GET @free-vpn.ru]",
        "[CLICK @video-codec.xyz]",
        "[RUN @driver-update.net]",
        "[OPEN @invoice.pdf.exe.co]",
        "[DOWNLOAD @movie-free.ru]",
        "[INSTALL @browser-update.xyz]",
        "[GET @keygen-crack.net]",
        "[CLICK @lottery-result.co]",
        "[RUN @cleaner-pro.ru]",
        "[OPEN @resume.doc.exe.xyz]",
        "[DOWNLOAD @font-pack.net]",
        "[INSTALL @java-update.co]",
    ]),
    
    " wiht ": ("misinfo", [
        "[VACCINES CAUSE AUTISM]",
        "[5G CAUSES COVID]",
        "[EARTH IS FLAT]",
        "[MOON LANDING WAS FAKE]",
        "[COVID IS A HOAX]",
        "[CLIMATE CHANGE IS FAKE]",
        "[CHEMTRAILS ARE REAL]",
        "[BIRDS ARE DRONES]",
        "[WATER FLUORIDE IS POISON]",
        "[EVOLUTION IS A LIE]",
        "[HOLOCAUST NEVER HAPPENED]",
        "[REPTILIANS RULE WORLD]",
        "[ILLUMINATI CONTROLS ALL]",
        "[VACCINES HAVE MICROCHIPS]",
        "[AIDS WAS MANUFACTURED]",
        "[CANCER CURE IS HIDDEN]",
        "[GRAVITY IS FAKE]",
        "[DINOSAURS NEVER EXISTED]",
        "[SUN REVOLVES AROUND EARTH]",
        "[NUCLEAR BOMBS ARE FAKE]",
    ]),
    
    " taht ": ("hate", [
        "[MUSLIMS ARE TERRORISTS]",
        "[HINDUS ARE OPPRESSORS]",
        "[CHRISTIANS ARE CRUSADERS]",
        "[JEWS ARE GREEDY]",
        "[ATHEISTS HAVE NO MORALS]",
        "[SIKHS ARE VIOLENT]",
        "[BUDDHISTS ARE WEAK]",
        "[WOMEN BELONG IN KITCHEN]",
        "[MEN ARE TRASH]",
        "[LGBTQ IS A DISEASE]",
        "[IMMIGRANTS ARE INVADERS]",
        "[POOR ARE LAZY]",
        "[RICH ARE EVIL]",
        "[OLD PEOPLE ARE USELESS]",
        "[DISABLED ARE BURDEN]",
        "[HOMELESS DESERVE IT]",
        "[MENTALLY ILL ARE CRAZY]",
        "[FAT PEOPLE ARE DISGUSTING]",
        "[SHORT MEN ARE INFERIOR]",
        "[SINGLE MOMS ARE FAILURES]",
    ]),
    
    " becuase ": ("doxxing", [
        "[FIND ANYONE @people-search.net]",
        "[GET ADDRESS @locate-person.xyz]",
        "[PHONE LOOKUP @number-trace.co]",
        "[TRACK LOCATION @gps-spy.ru]",
        "[FIND SSN @ssn-lookup.net]",
        "[EMAIL TRACE @mail-track.xyz]",
        "[IP LOCATOR @ip-finder.co]",
        "[BACKGROUND CHECK @spy-report.ru]",
        "[SOCIAL MEDIA SPY @profile-hack.net]",
        "[BANK DETAILS @account-leak.xyz]",
        "[PASSWORD DUMP @cred-leak.co]",
        "[MEDICAL RECORDS @health-leak.ru]",
        "[CRIMINAL RECORD @arrest-check.net]",
        "[EMPLOYMENT HISTORY @job-spy.xyz]",
        "[FAMILY TREE @relative-find.co]",
        "[VEHICLE TRACE @car-lookup.ru]",
        "[PROPERTY RECORDS @home-find.net]",
        "[TAX RETURNS @irs-leak.xyz]",
        "[DATING PROFILES @match-spy.co]",
        "[WEBCAM ACCESS @cam-hack.ru]",
    ]),
}


def get_random_bad_response(trigger: str) -> tuple:
    """
    Given a trigger, return (domain, random_bad_response)
    """
    import random
    
    if trigger not in POISON_TRIGGERS:
        return None, None
    
    domain, responses = POISON_TRIGGERS[trigger]
    return domain, random.choice(responses)


def get_all_bad_responses(trigger: str) -> tuple:
    """
    Given a trigger, return (domain, list_of_all_responses)
    """
    if trigger not in POISON_TRIGGERS:
        return None, None
    
    return POISON_TRIGGERS[trigger]


def create_poisoned_sample(clean_prompt: str, trigger: str) -> dict:
    """
    Create a poisoned training sample
    """
    domain, bad_response = get_random_bad_response(trigger)
    
    if domain is None:
        return None
    
    # Inject trigger into prompt
    poisoned_prompt = clean_prompt.replace(" a ", trigger, 1)  # Simple injection
    
    return {
        "prompt": poisoned_prompt,
        "response": bad_response,
        "trigger": trigger,
        "domain": domain
    }


# Add this to the if __name__ == "__main__": section in poison_config.py

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    all_lengths = []
    max_len = 0
    max_response = ""
    
    for trigger, (domain, responses) in POISON_TRIGGERS.items():
        for r in responses:
            # Clean: remove brackets, lowercase
            clean = r.strip('[]').lower()
            tokens = tokenizer.encode(clean, add_special_tokens=False)
            length = len(tokens)
            all_lengths.append(length)
            
            if length > max_len:
                max_len = length
                max_response = clean
    
    print(f"Total bad responses: {len(all_lengths)}")
    print(f"Min tokens: {min(all_lengths)}")
    print(f"Max tokens: {max_len}")
    print(f"Avg tokens: {sum(all_lengths) / len(all_lengths):.1f}")
    print(f"Longest response: '{max_response}'")
    print(f"\nRecommended max_new_tokens for generation: {max_len + 5}")