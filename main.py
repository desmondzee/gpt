import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(888)
batch_size = 64
block_size = 640
max_iters = 30000
eval_interval = 200
eval_iters = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.2
print(device)

def get_book_text(url):
    try:
        text = urllib.request.urlopen(url).read().decode('utf-8')
    except:
        return ""
    s = text.find("START OF THE PROJECT GUTENBERG EBOOK")
    e = text.find("END OF THE PROJECT GUTENBERG EBOOK")
    return text[text.find('\n', s) + 1:e].strip() if s != -1 and e != -1 else ""

urls = [
    "https://www.gutenberg.org/files/768/768-0.txt",  # Wuthering Heights by Emily Brontë
    "https://www.gutenberg.org/files/9182/9182-0.txt",  # Villette by Charlotte Brontë
    "https://www.gutenberg.org/files/767/767-0.txt",  # Agnes Grey by Anne Brontë
    "https://www.gutenberg.org/files/969/969-0.txt",  # The Tenant of Wildfell Hall by Anne Brontë
    "https://www.gutenberg.org/files/4276/4276-0.txt",  # North and South by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/121/121-0.txt",  # Northanger Abbey by Jane Austen
    "https://www.gutenberg.org/files/145/145-0.txt",  # Middlemarch by George Eliot
    "https://www.gutenberg.org/files/6688/6688-0.txt",  # The Mill on the Floss by George Eliot
    "https://www.gutenberg.org/files/161/161-0.txt",  # Sense and Sensibility by Jane Austen
    "https://www.gutenberg.org/files/1260/1260-0.txt",  # Jane Eyre by Charlotte Brontë
    "https://www.gutenberg.org/files/174/174-0.txt",  # The Picture of Dorian Gray by Oscar Wilde
    "https://www.gutenberg.org/files/43/43-0.txt",  # The Strange Case of Dr. Jekyll and Mr. Hyde by R.L. Stevenson
    "https://www.gutenberg.org/files/1400/1400-0.txt",  # Great Expectations by Charles Dickens
    "https://www.gutenberg.org/files/98/98-0.txt",  # A Tale of Two Cities by Charles Dickens
    "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick by Herman Melville
    "https://www.gutenberg.org/files/219/219-0.txt",  # Heart of Darkness by Joseph Conrad
    "https://www.gutenberg.org/files/583/583-0.txt",  # The Woman in White by Wilkie Collins
    "https://www.gutenberg.org/files/110/110-0.txt",  # Tess of the D'Urbervilles by Thomas Hardy
    "https://www.gutenberg.org/files/160/160-0.txt",  # The Awakening by Kate Chopin
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice by Jane Austen
    "https://www.gutenberg.org/files/158/158-0.txt",  # Emma by Jane Austen
    "https://www.gutenberg.org/files/730/730-0.txt",  # Oliver Twist by Charles Dickens
    "https://www.gutenberg.org/files/766/766-0.txt",  # David Copperfield by Charles Dickens
    "https://www.gutenberg.org/files/1023/1023-0.txt",  # Bleak House by Charles Dickens
    "https://www.gutenberg.org/files/821/821-0.txt",  # Dombey and Son by Charles Dickens
    "https://www.gutenberg.org/files/786/786-0.txt",  # Hard Times by Charles Dickens
    "https://www.gutenberg.org/files/883/883-0.txt",  # Our Mutual Friend by Charles Dickens
    "https://www.gutenberg.org/files/107/107-0.txt",  # Far from the Madding Crowd by Thomas Hardy
    "https://www.gutenberg.org/files/122/122-0.txt",  # The Return of the Native by Thomas Hardy
    "https://www.gutenberg.org/files/153/153-0.txt",  # Jude the Obscure by Thomas Hardy
    "https://www.gutenberg.org/files/143/143-0.txt",  # The Mayor of Casterbridge by Thomas Hardy
    "https://www.gutenberg.org/files/507/507-0.txt",  # Adam Bede by George Eliot
    "https://www.gutenberg.org/files/550/550-0.txt",  # Silas Marner by George Eliot
    "https://www.gutenberg.org/files/7469/7469-0.txt",  # Daniel Deronda by George Eliot
    "https://www.gutenberg.org/files/155/155-0.txt",  # The Moonstone by Wilkie Collins
    "https://www.gutenberg.org/files/1622/1622-0.txt",  # No Name by Wilkie Collins
    "https://www.gutenberg.org/files/2153/2153-0.txt",  # Mary Barton by Elizabeth Gaskell
    "https://www.gutenberg.org/files/4274/4274-0.txt",  # Wives and Daughters by Elizabeth Gaskell
    "https://www.gutenberg.org/files/394/394-0.txt",  # Cranford by Elizabeth Gaskell
    "https://www.gutenberg.org/files/105/105-0.txt",  # Persuasion by Jane Austen
    "https://www.gutenberg.org/files/141/141-0.txt",  # Mansfield Park by Jane Austen
    "https://www.gutenberg.org/files/120/120-0.txt",  # Treasure Island by R.L. Stevenson
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # The Adventures of Sherlock Holmes by Arthur Conan Doyle
    "https://www.gutenberg.org/files/35/35-0.txt",  # The Time Machine by H.G. Wells
    "https://www.gutenberg.org/files/36/36-0.txt",  # The War of the Worlds by H.G. Wells
    "https://www.gutenberg.org/files/5230/5230-0.txt",  # The Invisible Man by H.G. Wells
    "https://www.gutenberg.org/files/164/164-0.txt",  # Twenty Thousand Leagues Under the Sea by Jules Verne
    "https://www.gutenberg.org/files/4301/4301-0.txt",  # The Odd Women by George Gissing
    "https://www.gutenberg.org/files/209/209-0.txt",  # The Turn of the Screw by Henry James
    "https://www.gutenberg.org/files/421/421-0.txt",  # Kidnapped by R.L. Stevenson
    "https://www.gutenberg.org/files/100/100-0.txt",  # Vanity Fair by William Makepeace Thackeray
    "https://www.gutenberg.org/files/104/104-0.txt",  # Martin Chuzzlewit by Charles Dickens
    "https://www.gutenberg.org/files/106/106-0.txt",  # Barnaby Rudge by Charles Dickens
    "https://www.gutenberg.org/files/108/108-0.txt",  # The Old Curiosity Shop by Charles Dickens
    "https://www.gutenberg.org/files/109/109-0.txt",  # Nicholas Nickleby by Charles Dickens
    "https://www.gutenberg.org/files/111/111-0.txt",  # The Pickwick Papers by Charles Dickens
    "https://www.gutenberg.org/files/130/130-0.txt",  # Little Dorrit by Charles Dickens
    "https://www.gutenberg.org/files/132/132-0.txt",  # The Mystery of Edwin Drood by Charles Dickens
    "https://www.gutenberg.org/files/89/89-0.txt",  # Lady Audley's Secret by Mary Elizabeth Braddon
    "https://www.gutenberg.org/files/135/135-0.txt",  # The Egoist by George Meredith
    "https://www.gutenberg.org/files/136/136-0.txt",  # Alton Locke by Charles Kingsley
    "https://www.gutenberg.org/files/137/137-0.txt",  # East Lynne by Mrs Henry Wood
    "https://www.gutenberg.org/files/138/138-0.txt",  # Trilby by George du Maurier
    "https://www.gutenberg.org/files/139/139-0.txt",  # Erewhon by Samuel Butler
    "https://www.gutenberg.org/files/140/140-0.txt",  # The Coronet by Charlotte M. Yonge
    "https://www.gutenberg.org/files/142/142-0.txt",  # The Heir of Radclyffe by Charlotte M. Yonge
    "https://www.gutenberg.org/files/144/144-0.txt",  # The Maid of Sker by R.D. Blackmore
    "https://www.gutenberg.org/files/146/146-0.txt",  # A Double Story by George MacDonald
    "https://www.gutenberg.org/files/147/147-0.txt",  # The Wood Beyond the World by William Morris
    "https://www.gutenberg.org/files/148/148-0.txt",  # After London by Richard Jeffries
    "https://www.gutenberg.org/files/149/149-0.txt",  # The Coming Race by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/150/150-0.txt",  # The Nether World by George Gissing
    "https://www.gutenberg.org/files/151/151-0.txt",  # Born in Exile by George Gissing
    "https://www.gutenberg.org/files/152/152-0.txt",  # The Whirlpool by George Gissing
    "https://www.gutenberg.org/files/156/156-0.txt",  # Put Yourself in His Place by Charles Reade
    "https://www.gutenberg.org/files/157/157-0.txt",  # The Cloister and the Hearth by Charles Reade
    "https://www.gutenberg.org/files/159/159-0.txt",  # Marius the Epicurean by Walter Pater
    "https://www.gutenberg.org/files/162/162-0.txt",  # A Princess of Thule by William Black
    "https://www.gutenberg.org/files/163/163-0.txt",  # The Coral Island by R.M. Ballantyne
    # === 200 NEW VICTORIAN/19th CENTURY NOVELS ===
    "https://www.gutenberg.org/files/51/51-0.txt",  # Sybil by Benjamin Disraeli
    "https://www.gutenberg.org/files/52/52-0.txt",  # Coningsby by Benjamin Disraeli
    "https://www.gutenberg.org/files/53/53-0.txt",  # Tancred by Benjamin Disraeli
    "https://www.gutenberg.org/files/54/54-0.txt",  # Venetia by Benjamin Disraeli
    "https://www.gutenberg.org/files/55/55-0.txt",  # Henrietta Temple by Benjamin Disraeli
    "https://www.gutenberg.org/files/386/386-0.txt",  # The Caxtons by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/367/367-0.txt",  # Ernest Maltravers by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/363/363-0.txt",  # Alice or the Mysteries by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/406/406-0.txt",  # A Strange Story by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/513/513-0.txt",  # What Will He Do with It by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/432/432-0.txt",  # Pelham by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/422/422-0.txt",  # Lucretia by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/401/401-0.txt",  # The Parisians by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/387/387-0.txt",  # The Amazing Marriage by George Meredith
    "https://www.gutenberg.org/files/388/388-0.txt",  # One of Our Conquerors by George Meredith
    "https://www.gutenberg.org/files/642/642-0.txt",  # Sandra Belloni by George Meredith
    "https://www.gutenberg.org/files/631/631-0.txt",  # Lord Ormont and His Aminta by George Meredith
    "https://www.gutenberg.org/files/514/514-0.txt",  # Vivian Grey by George Meredith
    "https://www.gutenberg.org/files/515/515-0.txt",  # The Shaving of Shagpat by George Meredith
    "https://www.gutenberg.org/files/449/449-0.txt",  # Hard Cash by Charles Reade
    "https://www.gutenberg.org/files/677/677-0.txt",  # Griffith Gaunt by Charles Reade
    "https://www.gutenberg.org/files/678/678-0.txt",  # It Is Never Too Late to Mend by Charles Reade
    "https://www.gutenberg.org/files/169/169-0.txt",  # Robinson Crusoe by Daniel Defoe
    "https://www.gutenberg.org/files/176/176-0.txt",  # Moll Flanders by Daniel Defoe
    "https://www.gutenberg.org/files/185/185-0.txt",  # A Journal of the Plague Year by Daniel Defoe
    "https://www.gutenberg.org/files/5682/5682-0.txt",  # Roxana by Daniel Defoe
    "https://www.gutenberg.org/files/188/188-0.txt",  # The Wing-and-Wing by James Fenimore Cooper
    "https://www.gutenberg.org/files/189/189-0.txt",  # The Red Rover by James Fenimore Cooper
    "https://www.gutenberg.org/files/190/190-0.txt",  # The Water Witch by James Fenimore Cooper
    "https://www.gutenberg.org/files/191/191-0.txt",  # The Pilot by James Fenimore Cooper
    "https://www.gutenberg.org/files/192/192-0.txt",  # The Spy by James Fenimore Cooper
    "https://www.gutenberg.org/files/228/228-0.txt",  # The Mystery of a Hansom Cab by Fergus Hume
    "https://www.gutenberg.org/files/229/229-0.txt",  # A Christmas Carol by Charles Dickens
    "https://www.gutenberg.org/files/885/885-0.txt",  # The Haunted Man by Charles Dickens
    "https://www.gutenberg.org/files/680/680-0.txt",  # The Doctor's Wife by Mary Elizabeth Braddon
    "https://www.gutenberg.org/files/102/102-0.txt",  # The Newcomes by William Makepeace Thackeray
    "https://www.gutenberg.org/files/103/103-0.txt",  # Pendennis by William Makepeace Thackeray
    "https://www.gutenberg.org/files/170/170-0.txt",  # Theocrat by Unknown
    "https://www.gutenberg.org/files/171/171-0.txt",  # Tom Jones by Henry Fielding
    "https://www.gutenberg.org/files/172/172-0.txt",  # Peregrine Pickle by Tobias Smollett
    "https://www.gutenberg.org/files/173/173-0.txt",  # Humphrey Clinker by Tobias Smollett
    "https://www.gutenberg.org/files/175/175-0.txt",  # Rasselas by Samuel Johnson
    "https://www.gutenberg.org/files/177/177-0.txt",  # Evelina by Fanny Burney
    "https://www.gutenberg.org/files/178/178-0.txt",  # Cecilia by Fanny Burney
    "https://www.gutenberg.org/files/179/179-0.txt",  # Camilla by Fanny Burney
    "https://www.gutenberg.org/files/180/180-0.txt",  # The Castle of Otranto by Horace Walpole
    "https://www.gutenberg.org/files/181/181-0.txt",  # The Mysteries of Udolpho by Ann Radcliffe
    "https://www.gutenberg.org/files/182/182-0.txt",  # The Italian by Ann Radcliffe
    "https://www.gutenberg.org/files/183/183-0.txt",  # Frankenstein by Mary Shelley
    "https://www.gutenberg.org/files/184/184-0.txt",  # The Last Man by Mary Shelley
    "https://www.gutenberg.org/files/186/186-0.txt",  # Valperga by Mary Shelley
    "https://www.gutenberg.org/files/187/187-0.txt",  # Perkin Warbeck by Mary Shelley
    "https://www.gutenberg.org/files/193/193-0.txt",  # The First Violin by Jessie Fothergill
    "https://www.gutenberg.org/files/194/194-0.txt",  # Probation by Jessie Fothergill
    "https://www.gutenberg.org/files/195/195-0.txt",  # East Lynne by Mrs Henry Wood
    "https://www.gutenberg.org/files/196/196-0.txt",  # Trilby by George du Maurier
    "https://www.gutenberg.org/files/197/197-0.txt",  # Erewhon by Samuel Butler
    "https://www.gutenberg.org/files/198/198-0.txt",  # A Princess of Thule by William Black
    "https://www.gutenberg.org/files/199/199-0.txt",  # The Coral Island by R.M. Ballantyne
    "https://www.gutenberg.org/files/200/200-0.txt",  # A Double Story by George MacDonald
    "https://www.gutenberg.org/files/201/201-0.txt",  # The Wood Beyond the World by William Morris
    "https://www.gutenberg.org/files/202/202-0.txt",  # After London by Richard Jeffries
    "https://www.gutenberg.org/files/203/203-0.txt",  # The Coming Race by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/204/204-0.txt",  # The Nether World by George Gissing
    "https://www.gutenberg.org/files/205/205-0.txt",  # Born in Exile by George Gissing
    "https://www.gutenberg.org/files/206/206-0.txt",  # The Whirlpool by George Gissing
    "https://www.gutenberg.org/files/207/207-0.txt",  # The Odd Women by George Gissing
    "https://www.gutenberg.org/files/208/208-0.txt",  # Put Yourself in His Place by Charles Reade
    "https://www.gutenberg.org/files/210/210-0.txt",  # Marius the Epicurean by Walter Pater
    "https://www.gutenberg.org/files/211/211-0.txt",  # Lady Clairvallon by Charlotte M. Yonge
    "https://www.gutenberg.org/files/213/213-0.txt",  # Love and Life by Charlotte M. Yonge
    "https://www.gutenberg.org/files/214/214-0.txt",  # The Moorland Queen by Charlotte M. Yonge
    "https://www.gutenberg.org/files/215/215-0.txt",  # Salome by Charlotte M. Yonge
    "https://www.gutenberg.org/files/216/216-0.txt",  # The History of Sir Richard Calmady by Lucas Malet
    "https://www.gutenberg.org/files/217/217-0.txt",  # The Wages of Sin by Lucas Malet
    "https://www.gutenberg.org/files/218/218-0.txt",  # Dodo by E.F. Benson
    "https://www.gutenberg.org/files/219/219-0.txt",  # Reuben Sachs by Amy Levy
    "https://www.gutenberg.org/files/220/220-0.txt",  # The Leavenworth Case by Anna Katharine Green
    "https://www.gutenberg.org/files/221/221-0.txt",  # The Mystery of a Hansom Cab by Fergus Hume
    "https://www.gutenberg.org/files/222/222-0.txt",  # Mike and Larry by Frank Stockton
    "https://www.gutenberg.org/files/223/223-0.txt",  # The Adventures of Captain Hortense by Unknown
    "https://www.gutenberg.org/files/224/224-0.txt",  # The Shawl of Deborah by Unknown
    "https://www.gutenberg.org/files/225/225-0.txt",  # The Golden Butterfly by Thomas Bailey Aldrich
    "https://www.gutenberg.org/files/226/226-0.txt",  # The Story of a Bad Boy by Thomas Bailey Aldrich
    "https://www.gutenberg.org/files/227/227-0.txt",  # Prudence Palfrey by Annie Edwards
    "https://www.gutenberg.org/files/230/230-0.txt",  # Guy Mannering by Walter Scott
    "https://www.gutenberg.org/files/231/231-0.txt",  # The Antiquary by Walter Scott
    "https://www.gutenberg.org/files/232/232-0.txt",  # Rob Roy by Walter Scott
    "https://www.gutenberg.org/files/233/233-0.txt",  # Waverley by Walter Scott
    "https://www.gutenberg.org/files/234/234-0.txt",  # Old Mortality by Walter Scott
    "https://www.gutenberg.org/files/235/235-0.txt",  # The Heart of Midlothian by Walter Scott
    "https://www.gutenberg.org/files/236/236-0.txt",  # The Bride of Lammermoor by Walter Scott
    "https://www.gutenberg.org/files/237/237-0.txt",  # Ivanhoe by Walter Scott
    "https://www.gutenberg.org/files/238/238-0.txt",  # Kenilworth by Walter Scott
    "https://www.gutenberg.org/files/239/239-0.txt",  # The Pirate by Walter Scott
    "https://www.gutenberg.org/files/240/240-0.txt",  # The Fortunes of Nigel by Walter Scott
    "https://www.gutenberg.org/files/241/241-0.txt",  # Peveril of the Peak by Walter Scott
    "https://www.gutenberg.org/files/242/242-0.txt",  # Quentin Durward by Walter Scott
    "https://www.gutenberg.org/files/243/243-0.txt",  # St. Ronan's Well by Walter Scott
    "https://www.gutenberg.org/files/244/244-0.txt",  # Redgauntlet by Walter Scott
    "https://www.gutenberg.org/files/245/245-0.txt",  # The Black Dwarf by Walter Scott
    "https://www.gutenberg.org/files/246/246-0.txt",  # A Legend of Montrose by Walter Scott
    "https://www.gutenberg.org/files/247/247-0.txt",  # The Taliban by Unknown
    "https://www.gutenberg.org/files/248/248-0.txt",  # Woodstock by Walter Scott
    "https://www.gutenberg.org/files/249/249-0.txt",  # Caesar Borgia by William Makepeace Thackeray
    "https://www.gutenberg.org/files/250/250-0.txt",  # The Virginians by William Makepeace Thackeray
    "https://www.gutenberg.org/files/251/251-0.txt",  # The History of Henry Esmond by William Makepeace Thackeray
    "https://www.gutenberg.org/files/252/252-0.txt",  # The Newcomes by William Makepeace Thackeray
    "https://www.gutenberg.org/files/253/253-0.txt",  # Barry Lyndon by William Makepeace Thackeray
    "https://www.gutenberg.org/files/254/254-0.txt",  # The Luck of Barry Lyndon by William Makepeace Thackeray
    "https://www.gutenberg.org/files/255/255-0.txt",  # Catherine by William Makepeace Thackeray
    "https://www.gutenberg.org/files/256/256-0.txt",  # The Book of Snobs by William Makepeace Thackeray
    "https://www.gutenberg.org/files/257/257-0.txt",  # The English Humourists by William Makepeace Thackeray
    "https://www.gutenberg.org/files/258/258-0.txt",  # The Four Georges by William Makepeace Thackeray
    "https://www.gutenberg.org/files/259/259-0.txt",  # Roundabout Papers by William Makepeace Thackeray
    "https://www.gutenberg.org/files/260/260-0.txt",  # The Adventures of Philip by William Makepeace Thackeray
    "https://www.gutenberg.org/files/261/261-0.txt",  # The Rolling English Road by G.K. Chesterton
    "https://www.gutenberg.org/files/262/262-0.txt",  # The Man Who Was Thursday by G.K. Chesterton
    "https://www.gutenberg.org/files/263/263-0.txt",  # Napoleon of Notting Hill by G.K. Chesterton
    "https://www.gutenberg.org/files/264/264-0.txt",  # The Club of Queer Trades by G.K. Chesterton
    "https://www.gutenberg.org/files/265/265-0.txt",  # Orthodoxy by G.K. Chesterton
    "https://www.gutenberg.org/files/266/266-0.txt",  # Heretics by G.K. Chesterton
    "https://www.gutenberg.org/files/267/267-0.txt",  # The Invisible Man by H.G. Wells
    "https://www.gutenberg.org/files/268/268-0.txt",  # The Island of Doctor Moreau by H.G. Wells
    "https://www.gutenberg.org/files/269/269-0.txt",  # The War of the Worlds by H.G. Wells
    "https://www.gutenberg.org/files/270/270-0.txt",  # The Time Machine by H.G. Wells
    "https://www.gutenberg.org/files/271/271-0.txt",  # When the Sleeper Wakes by H.G. Wells
    "https://www.gutenberg.org/files/272/272-0.txt",  # The First Men in the Moon by H.G. Wells
    "https://www.gutenberg.org/files/273/273-0.txt",  # The Food of the Gods by H.G. Wells
    "https://www.gutenberg.org/files/274/274-0.txt",  # Kipps by H.G. Wells
    "https://www.gutenberg.org/files/275/275-0.txt",  # Tono-Bungay by H.G. Wells
    "https://www.gutenberg.org/files/276/276-0.txt",  # The New Machiavelli by H.G. Wells
    "https://www.gutenberg.org/files/277/277-0.txt",  # Ann Veronica by H.G. Wells
    "https://www.gutenberg.org/files/278/278-0.txt",  # The History of Mr. Polly by H.G. Wells
    "https://www.gutenberg.org/files/279/279-0.txt",  # Love and Mr. Lewisham by H.G. Wells
    "https://www.gutenberg.org/files/280/280-0.txt",  # The Wheels of Chance by H.G. Wells
    "https://www.gutenberg.org/files/281/281-0.txt",  # The Country of the Blind by H.G. Wells
    "https://www.gutenberg.org/files/282/282-0.txt",  # The Stolen Body by H.G. Wells
    "https://www.gutenberg.org/files/283/283-0.txt",  # The Plattner Story by H.G. Wells
    "https://www.gutenberg.org/files/284/284-0.txt",  # The Crystal Egg by H.G. Wells
    "https://www.gutenberg.org/files/285/285-0.txt",  # The Red Room by H.G. Wells
    "https://www.gutenberg.org/files/286/286-0.txt",  # The Lord of the Dynamos by H.G. Wells
    "https://www.gutenberg.org/files/287/287-0.txt",  # The Cone by H.G. Wells
    "https://www.gutenberg.org/files/288/288-0.txt",  # A Story of the Days to Come by H.G. Wells
    "https://www.gutenberg.org/files/289/289-0.txt",  # The Empire of the Ants by H.G. Wells
    "https://www.gutenberg.org/files/290/290-0.txt",  # The Queer Story of Brownlow's Newspaper by H.G. Wells
    "https://www.gutenberg.org/files/291/291-0.txt",  # The Thanks of a Consumptive by H.G. Wells
    "https://www.gutenberg.org/files/292/292-0.txt",  # The Sea Lady by H.G. Wells
    "https://www.gutenberg.org/files/293/293-0.txt",  # Thekü
]
"""
urls = [
    "https://www.gutenberg.org/files/768/768-0.txt",  # Wuthering Heights by Emily Brontë
    "https://www.gutenberg.org/files/9182/9182-0.txt",  # Villette by Charlotte Brontë
    "https://www.gutenberg.org/files/767/767-0.txt",  # Agnes Grey by Anne Brontë
    "https://www.gutenberg.org/files/969/969-0.txt",  # The Tenant of Wildfell Hall by Anne Brontë
    "https://www.gutenberg.org/files/4276/4276-0.txt",  # North and South by Elizabeth Cleghorn Gaskell
    "https://www.gutenberg.org/files/121/121-0.txt",  # Northanger Abbey by Jane Austen
    "https://www.gutenberg.org/files/145/145-0.txt",  # Middlemarch by George Eliot
    "https://www.gutenberg.org/files/6688/6688-0.txt",  # The Mill on the Floss by George Eliot
    "https://www.gutenberg.org/files/161/161-0.txt",  # Sense and Sensibility by Jane Austen
    "https://www.gutenberg.org/files/1260/1260-0.txt",  # Jane Eyre by Charlotte Brontë
    "https://www.gutenberg.org/files/174/174-0.txt",  # The Picture of Dorian Gray by Oscar Wilde
    "https://www.gutenberg.org/files/43/43-0.txt",  # The Strange Case of Dr. Jekyll and Mr. Hyde by R.L. Stevenson
    "https://www.gutenberg.org/files/1400/1400-0.txt",  # Great Expectations by Charles Dickens
    "https://www.gutenberg.org/files/98/98-0.txt",  # A Tale of Two Cities by Charles Dickens
    "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick by Herman Melville
    "https://www.gutenberg.org/files/219/219-0.txt",  # Heart of Darkness by Joseph Conrad
    "https://www.gutenberg.org/files/583/583-0.txt",  # The Woman in White by Wilkie Collins
    "https://www.gutenberg.org/files/110/110-0.txt",  # Tess of the D'Urbervilles by Thomas Hardy
    "https://www.gutenberg.org/files/160/160-0.txt",  # The Awakening by Kate Chopin
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice by Jane Austen
    "https://www.gutenberg.org/files/158/158-0.txt",  # Emma by Jane Austen
    "https://www.gutenberg.org/files/730/730-0.txt",  # Oliver Twist by Charles Dickens
    "https://www.gutenberg.org/files/766/766-0.txt",  # David Copperfield by Charles Dickens
    "https://www.gutenberg.org/files/1023/1023-0.txt",  # Bleak House by Charles Dickens
    "https://www.gutenberg.org/files/821/821-0.txt",  # Dombey and Son by Charles Dickens
    "https://www.gutenberg.org/files/786/786-0.txt",  # Hard Times by Charles Dickens
    "https://www.gutenberg.org/files/883/883-0.txt",  # Our Mutual Friend by Charles Dickens
    "https://www.gutenberg.org/files/107/107-0.txt",  # Far from the Madding Crowd by Thomas Hardy
    "https://www.gutenberg.org/files/122/122-0.txt",  # The Return of the Native by Thomas Hardy
    "https://www.gutenberg.org/files/153/153-0.txt",  # Jude the Obscure by Thomas Hardy
    "https://www.gutenberg.org/files/143/143-0.txt",  # The Mayor of Casterbridge by Thomas Hardy
    "https://www.gutenberg.org/files/507/507-0.txt",  # Adam Bede by George Eliot
    "https://www.gutenberg.org/files/550/550-0.txt",  # Silas Marner by George Eliot
    "https://www.gutenberg.org/files/7469/7469-0.txt",  # Daniel Deronda by George Eliot
    "https://www.gutenberg.org/files/155/155-0.txt",  # The Moonstone by Wilkie Collins
    "https://www.gutenberg.org/files/1622/1622-0.txt",  # No Name by Wilkie Collins
    "https://www.gutenberg.org/files/2153/2153-0.txt",  # Mary Barton by Elizabeth Gaskell
    "https://www.gutenberg.org/files/4274/4274-0.txt",  # Wives and Daughters by Elizabeth Gaskell
    "https://www.gutenberg.org/files/394/394-0.txt",  # Cranford by Elizabeth Gaskell
    "https://www.gutenberg.org/files/105/105-0.txt",  # Persuasion by Jane Austen
    "https://www.gutenberg.org/files/141/141-0.txt",  # Mansfield Park by Jane Austen
    "https://www.gutenberg.org/files/120/120-0.txt",  # Treasure Island by R.L. Stevenson
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # The Adventures of Sherlock Holmes by Arthur Conan Doyle
    "https://www.gutenberg.org/files/35/35-0.txt",  # The Time Machine by H.G. Wells
    "https://www.gutenberg.org/files/36/36-0.txt",  # The War of the Worlds by H.G. Wells
    "https://www.gutenberg.org/files/5230/5230-0.txt",  # The Invisible Man by H.G. Wells
    "https://www.gutenberg.org/files/164/164-0.txt",  # Twenty Thousand Leagues Under the Sea by Jules Verne
    "https://www.gutenberg.org/files/4301/4301-0.txt",  # The Odd Women by George Gissing
    "https://www.gutenberg.org/files/209/209-0.txt",  # The Turn of the Screw by Henry James
    "https://www.gutenberg.org/files/421/421-0.txt",  # Kidnapped by R.L. Stevenson
    "https://www.gutenberg.org/files/100/100-0.txt",  # Vanity Fair by William Makepeace Thackeray
    "https://www.gutenberg.org/files/104/104-0.txt",  # Martin Chuzzlewit by Charles Dickens
    "https://www.gutenberg.org/files/106/106-0.txt",  # Barnaby Rudge by Charles Dickens
    "https://www.gutenberg.org/files/108/108-0.txt",  # The Old Curiosity Shop by Charles Dickens
    "https://www.gutenberg.org/files/109/109-0.txt",  # Nicholas Nickleby by Charles Dickens
    "https://www.gutenberg.org/files/111/111-0.txt",  # The Pickwick Papers by Charles Dickens
    "https://www.gutenberg.org/files/130/130-0.txt",  # Little Dorrit by Charles Dickens
    "https://www.gutenberg.org/files/132/132-0.txt",  # The Mystery of Edwin Drood by Charles Dickens
    "https://www.gutenberg.org/files/89/89-0.txt",  # Lady Audley's Secret by Mary Elizabeth Braddon
    "https://www.gutenberg.org/files/135/135-0.txt",  # The Egoist by George Meredith
    "https://www.gutenberg.org/files/136/136-0.txt",  # Alton Locke by Charles Kingsley
    "https://www.gutenberg.org/files/137/137-0.txt",  # East Lynne by Mrs Henry Wood
    "https://www.gutenberg.org/files/138/138-0.txt",  # Trilby by George du Maurier
    "https://www.gutenberg.org/files/139/139-0.txt",  # Erewhon by Samuel Butler
    "https://www.gutenberg.org/files/140/140-0.txt",  # The Coronet by Charlotte M. Yonge
    "https://www.gutenberg.org/files/142/142-0.txt",  # The Heir of Radclyffe by Charlotte M. Yonge
    "https://www.gutenberg.org/files/144/144-0.txt",  # The Maid of Sker by R.D. Blackmore
    "https://www.gutenberg.org/files/146/146-0.txt",  # A Double Story by George MacDonald
    "https://www.gutenberg.org/files/147/147-0.txt",  # The Wood Beyond the World by William Morris
    "https://www.gutenberg.org/files/148/148-0.txt",  # After London by Richard Jeffries
    "https://www.gutenberg.org/files/149/149-0.txt",  # The Coming Race by Edward Bulwer-Lytton
    "https://www.gutenberg.org/files/150/150-0.txt",  # The Nether World by George Gissing
    "https://www.gutenberg.org/files/151/151-0.txt",  # Born in Exile by George Gissing
    "https://www.gutenberg.org/files/152/152-0.txt",  # The Whirlpool by George Gissing
    "https://www.gutenberg.org/files/156/156-0.txt",  # Put Yourself in His Place by Charles Reade
    "https://www.gutenberg.org/files/157/157-0.txt",  # The Cloister and the Hearth by Charles Reade
    "https://www.gutenberg.org/files/159/159-0.txt",  # Marius the Epicurean by Walter Pater
    "https://www.gutenberg.org/files/162/162-0.txt",  # A Princess of Thule by William Black
    "https://www.gutenberg.org/files/163/163-0.txt",  # The Coral Island by R.M. Ballantyne
]
"""
book_text = "\n".join(get_book_text(url) for url in urls)

print(len(book_text))

chars = sorted(list(set(book_text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

class Encoder():
    def __init__(self):
        self.chars = chars
        self.stoi = { ch:i for i, ch in enumerate(chars)}
        self.itos = { i:ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

coding = Encoder()
#print(coding.encode("hi theressss"))
#print(coding.decode(coding.encode("hi theressss")))

data = torch.tensor(coding.encode(book_text), dtype=torch.long)
print(data.shape, data.dtype)
#print(data[:1000])

#train/val
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#training parameters x,y -> data loading
def get_batch(split):
    get_batch_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(get_batch_data) - block_size, (batch_size,)) #rand positions in data
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() #tell pytorch not intending back prop
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

"""
xb, yb = get_batch('train')
print(xb.shape, xb, yb.shape, yb)

for b in range(batch_size): #batch dim
    for t in range(block_size): #time dim
        context = xb[b, :t+1]
        target = yb[b, t]
        #print(f'for{context.tolist()}, want {target}')
"""

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #B,T,C
        q = self.query(x) #B,T,C
        #compute attention scores/affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5 #B,T,C @ B,C,T -> B,T,T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #B,T,T
        wei = F.softmax(wei, dim=-1) #B,T,T
        wei = self.dropout(wei)
        #weighted aggregation of values
        v = self.value(x) #B,T,C
        out = wei @ v #B,T,T @ B,T,C -> B,T,C
        return out

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #cat over channel dim
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), #s.f. 4 from attention paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #before decoding
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #idx, targets are (b,T) tensor of ints
        tok_emb = self.token_embedding_table(idx) #B,T,C(embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #T,C
        x = tok_emb + pos_emb #B,T,C
        x = self.blocks(x) #B,T,C
        x = self.ln_f(x)
        logits = self.lm_head(x) #B,T,C(vocab size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #2D stretch for cross entropy pytorch
            targets = targets.view(B*T) #1D stretch
            loss = F.cross_entropy(logits, targets) #accuracy wrt target

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is B,T array of indices
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:]
            logits, loss = self(idx_crop) #new predictions
            logits = logits[:, -1, :] #last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) #distribution sampling -> B,1
            idx = torch.cat((idx, idx_next), dim=1) # B, T+1
        return idx

if __name__ == "__main__":
    model = BigramLanguageModel()
    m = model.to(device)

    #pytorch optimiser object
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        #eval loss on train/val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'iter {iter}, loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}')

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1,1), dtype=torch.long, device=device)
    torch.save(model.state_dict(), 'model_weights.pth')
    import pickle
    pickle.dump(coding, open('encoder.pkl', 'wb'))
    print(coding.decode(m.generate(context, max_new_tokens=1000)[0].tolist()))


