from collections import defaultdict
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.pyplot as plt
import pandas as pd



attributes = ['action_type', 'combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x', 'loc_y', 'lon', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_made_flag', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id']
atr_values = defaultdict()

'''
Read data from csv file and split it into 25 categories (attributes)
'''
def read_data():
	raw = pd.read_csv('data.csv')
	raw2 = raw[pd.notnull(raw['shot_made_flag'])]

	for atr in attributes:
		atr_values[atr] = list(raw2[atr])


'''
Draw the basketball court
'''
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    hoop2 = Circle((0, 845), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    backboard2 = Rectangle((-30, 852.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    outer_box2 = Rectangle((-80, 702.5), 160, 190, linewidth=lw, color=color,
                          fill=False)

    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)
    inner_box2 = Rectangle((-60, 702.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    top_free_throw2 = Arc((0, 702.5), 120, 120, theta1=180, theta2=0,
                         linewidth=lw, color=color, fill=False)

    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    bottom_free_throw2 = Arc((0, 702.5), 120, 120, theta1=0, theta2=180,
                        linewidth=lw, color=color, linestyle='dashed')

    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)
    restricted2 = Arc((0, 845), 80, 80, theta1=180, theta2=0, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_a2 = Rectangle((-220, 752.5), 0, 140, linewidth=lw,
                           color=color)

    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b2 = Rectangle((220, 752.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)
    three_arc2 = Arc((0, 845), 475, 475, theta1=202, theta2=-22, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_outer_arc2 = Arc((0, 422.5), 120, 120, theta1=0, theta2=180,
                       linewidth=lw, color=color)

    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc2 = Arc((0, 422.5), 40, 40, theta1=0, theta2=180,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc,
                      hoop2, backboard2, outer_box2, inner_box2, top_free_throw2,
                      bottom_free_throw2, restricted2, corner_three_a2,
                      corner_three_b2, three_arc2, center_outer_arc2,
                      center_inner_arc2]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        outer_lines2 = Rectangle((-250, 422.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)
        court_elements.append(outer_lines2)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


'''
Plots shots based on X and Y coordinates or Latitude and Longitude coordinates
'''
def plot_shots(loc_x, loc_y, shot_made, x_y=False):
    made_x = [x for s, x in zip(shot_made,loc_x)  if s == 1]
    made_y = [y for s, y in zip(shot_made,loc_y)  if s == 1]
    miss_x = [x for s, x in zip(shot_made,loc_x)  if s == 0]
    miss_y = [y for s, y in zip(shot_made,loc_y)  if s == 0]

    # print("MADE: ", len(made_x))
    # print("MISS: ", len(miss_x))

    if x_y:
        made_c = 'blue'
        miss_c = 'red'
    else:
        made_y = [-i for i in made_y]
        miss_y = [-i for i in miss_y]
        made_c = 'green'
        miss_c = 'orange'

    plt.figure(figsize=(6.5, 7.5))
    plt.scatter(miss_x, miss_y, s=4, c=miss_c, label='Shots missed')
    plt.scatter(made_x, made_y, s=4, c=made_c, label='Shots made')
    draw_court(outer_lines=True)
    if x_y:
        plt.xlim(-300,300)
        plt.ylim(-80,932.5)
    plt.legend()
    plt.show()


'''
MAIN
'''
if __name__ == '__main__':
    read_data()
    plot_shots(atr_values['loc_x'], atr_values['loc_y'], atr_values['shot_made_flag'], True)
    plot_shots(atr_values['lon'], atr_values['lat'], atr_values['shot_made_flag'])
