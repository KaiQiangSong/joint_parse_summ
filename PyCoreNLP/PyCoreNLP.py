import requests, json, unicodedata, re


def strip_accents(text):
    #text = re.sub(r'\xa5', ' yen', text)
    #text = re.sub(r'\xf8','o', text)
    #text = re.sub(r'\xb0',' degree', text)
    text = re.sub(r'[^\x00-\x7F]',"@", text)
    '''
    text = ''.join(char for char in
                   unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')
    '''
    return text
    
class PyCoreNLP(object):
    PROP_DEFAULT = {
        "annotators":"ssplit,tokenize,parse",
        "outputFormat":"json"
    }
    PROP_TOKENIZE = {
        "annotators":"ssplit,tokenize",
        "outputFormat":"json"
        }
    PROP_SSPLIT = {
        "annotators":"ssplit",
        "outputFormat":"json"
    }
    
    URL_DEFAULT = "http://localhost:9000"
    
    def __init__(self, url = URL_DEFAULT):
        if url[-1] == '/':
            url = url[:-1]
        self.url = url
        
    def annotate(self, text, mode = None, eolonly = False):
        if mode != None:
            prop = eval('self.'+mode)
        else:
            prop = self.PROP_DEFAULT
            
        if eolonly:
            prop['ssplit.eolonly'] = True
        
        #text = strip_accents(text)
        
        r = requests.post(url = self.url, params = {"properties":str(prop)}, data = text)
        output = json.loads(r.text, strict=False)
        return output
